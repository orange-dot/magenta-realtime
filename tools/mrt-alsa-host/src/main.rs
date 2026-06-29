use std::env;
use std::fs::OpenOptions;
use std::os::unix::fs::FileExt;
use std::path::PathBuf;
use std::thread;
use std::time::{Duration, Instant};

use alsa::pcm::{Access, Format, Frames, HwParams, PCM};
use alsa::{Direction, ValueOr};
use mrt_alsa_host::{
    DEFAULT_CHANNELS, DEFAULT_SAMPLE_RATE, PlaybackFormat, RING_BYTES_PER_FRAME, RING_HEADER_SIZE,
    RingHeader, auto_playback_format_order, convert_s16_interleaved_to_playback_format,
    parse_alsa_hw_device, ring_data_offset_for_frame, ring_file_size,
};

const DEFAULT_RING_PATH: &str = "/tmp/mrt2-pc4ms-live.ring";
const DEFAULT_DEVICE: &str = "hw:CARD=AG06AG03,DEV=0";

#[derive(Debug, Clone)]
struct Args {
    ring: PathBuf,
    device: String,
    format: FormatSelection,
    period_frames: usize,
    alsa_buffer_periods: usize,
    duration_seconds: Option<u64>,
    stats_every_ms: u64,
    start_threshold_frames: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FormatSelection {
    Auto,
    Exact(PlaybackFormat),
}

#[derive(Debug)]
struct HostMetrics {
    started: Instant,
    played_frames: u64,
    ring_underrun_frames: u64,
    alsa_xruns: u64,
    zero_fill_events: u64,
    low_water_frames: u64,
}

struct PlaybackDevice {
    pcm: PCM,
    format: PlaybackFormat,
    period_frames: usize,
    buffer_frames: usize,
}

impl HostMetrics {
    fn new(header: &RingHeader) -> Self {
        Self {
            started: Instant::now(),
            played_frames: 0,
            ring_underrun_frames: 0,
            alsa_xruns: 0,
            zero_fill_events: 0,
            low_water_frames: header.low_water_frames,
        }
    }
}

fn main() {
    if let Err(err) = run(parse_args()) {
        eprintln!("mrt-alsa-host: {err}");
        std::process::exit(1);
    }
}

fn run(args: Args) -> Result<(), String> {
    parse_alsa_hw_device(&args.device)?;
    let ring = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&args.ring)
        .map_err(|err| format!("failed to open ring {}: {err}", args.ring.display()))?;
    let header = read_ring_header(&ring)?;
    let ring_len = ring
        .metadata()
        .map_err(|err| format!("failed to stat ring {}: {err}", args.ring.display()))?
        .len() as usize;
    if ring_len < ring_file_size(&header) {
        return Err(format!(
            "ring file {} is too small: {} < {}",
            args.ring.display(),
            ring_len,
            ring_file_size(&header)
        ));
    }
    let header = wait_for_start_threshold(&ring, &args, header)?;
    let playback = open_playback_auto(&args)?;
    let io = playback.pcm.io_bytes();
    let mut metrics = HostMetrics::new(&header);
    let mut next_stats = Instant::now() + Duration::from_millis(args.stats_every_ms);
    let stop_at = args
        .duration_seconds
        .map(|seconds| Instant::now() + Duration::from_secs(seconds));

    println!(
        "{{\"schema\":\"mrt_alsa_host.start.v1\",\"ring\":\"{}\",\"device\":\"{}\",\"format\":\"{}\",\"sample_rate\":{},\"channels\":{},\"period_frames\":{},\"alsa_buffer_frames\":{},\"alsa_buffer_periods\":{},\"start_threshold_frames\":{}}}",
        args.ring.display(),
        args.device,
        playback.format.label(),
        DEFAULT_SAMPLE_RATE,
        DEFAULT_CHANNELS,
        playback.period_frames,
        playback.buffer_frames,
        args.alsa_buffer_periods,
        args.start_threshold_frames
    );

    loop {
        if stop_at.is_some_and(|deadline| Instant::now() >= deadline) {
            break;
        }
        let mut header = read_ring_header(&ring)?;
        let (samples, readable, missing) =
            read_s16_period_or_zero_fill(&ring, &header, playback.period_frames)?;
        if missing > 0 {
            metrics.zero_fill_events = metrics.zero_fill_events.saturating_add(1);
            metrics.ring_underrun_frames = metrics.ring_underrun_frames.saturating_add(missing);
            header.underrun_frames = header.underrun_frames.saturating_add(missing);
        }
        header.read_cursor = header.read_cursor.saturating_add(readable);
        let available_after = header.write_cursor.saturating_sub(header.read_cursor);
        header.low_water_frames = header.low_water_frames.min(available_after);
        metrics.low_water_frames = metrics.low_water_frames.min(available_after);
        write_ring_header(&ring, &header)?;

        let payload = convert_s16_interleaved_to_playback_format(&samples, playback.format);
        write_period(
            &playback.pcm,
            &io,
            &payload,
            playback.period_frames,
            &mut metrics,
        )?;
        metrics.played_frames = metrics
            .played_frames
            .saturating_add(playback.period_frames as u64);

        if Instant::now() >= next_stats {
            print_stats(&args, &playback, &metrics, &read_ring_header(&ring)?);
            next_stats = Instant::now() + Duration::from_millis(args.stats_every_ms);
        }
    }

    let _ = playback.pcm.drain();
    print_stats(&args, &playback, &metrics, &read_ring_header(&ring)?);
    Ok(())
}

fn parse_args() -> Args {
    let mut args = Args {
        ring: PathBuf::from(DEFAULT_RING_PATH),
        device: DEFAULT_DEVICE.to_string(),
        format: FormatSelection::Auto,
        period_frames: 960,
        alsa_buffer_periods: 4,
        duration_seconds: None,
        stats_every_ms: 1_000,
        start_threshold_frames: 0,
    };
    let mut iter = env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--ring" => args.ring = PathBuf::from(require_value(&arg, iter.next())),
            "--device" => args.device = require_value(&arg, iter.next()),
            "--format" => {
                let value = require_value(&arg, iter.next());
                args.format = if value.eq_ignore_ascii_case("auto") {
                    FormatSelection::Auto
                } else {
                    FormatSelection::Exact(
                        PlaybackFormat::parse(&value).unwrap_or_else(|err| die(&err)),
                    )
                };
            }
            "--period-frames" => {
                args.period_frames = require_value(&arg, iter.next())
                    .parse()
                    .unwrap_or_else(|_| die("--period-frames must be a positive integer"));
            }
            "--alsa-buffer-periods" => {
                args.alsa_buffer_periods = require_value(&arg, iter.next())
                    .parse()
                    .unwrap_or_else(|_| die("--alsa-buffer-periods must be a positive integer"));
            }
            "--duration-seconds" => {
                args.duration_seconds = Some(
                    require_value(&arg, iter.next())
                        .parse()
                        .unwrap_or_else(|_| die("--duration-seconds must be an integer")),
                );
            }
            "--stats-every-ms" => {
                args.stats_every_ms = require_value(&arg, iter.next())
                    .parse()
                    .unwrap_or_else(|_| die("--stats-every-ms must be an integer"));
            }
            "--start-threshold-frames" => {
                args.start_threshold_frames = require_value(&arg, iter.next())
                    .parse()
                    .unwrap_or_else(|_| die("--start-threshold-frames must be an integer"));
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            unknown => die(&format!("unknown argument {unknown:?}")),
        }
    }
    if args.period_frames == 0 {
        die("--period-frames must be greater than zero");
    }
    if args.alsa_buffer_periods == 0 {
        die("--alsa-buffer-periods must be greater than zero");
    }
    if args.stats_every_ms == 0 {
        die("--stats-every-ms must be greater than zero");
    }
    args
}

fn print_help() {
    println!(
        "Usage: mrt-alsa-host [--ring PATH] [--device hw:CARD=AG06AG03,DEV=0] \\
         [--format auto|s32_le|s24_le|s24_3le|s16_le] [--duration-seconds N] \\
         [--period-frames N] [--alsa-buffer-periods N] [--start-threshold-frames N]"
    );
}

fn require_value(flag: &str, value: Option<String>) -> String {
    value.unwrap_or_else(|| die(&format!("{flag} requires a value")))
}

fn die(message: &str) -> ! {
    eprintln!("mrt-alsa-host: {message}");
    std::process::exit(2);
}

fn open_playback_auto(args: &Args) -> Result<PlaybackDevice, String> {
    match args.format {
        FormatSelection::Exact(format) => open_playback_device(args, format),
        FormatSelection::Auto => {
            let mut failures = Vec::new();
            for format in auto_playback_format_order() {
                match open_playback_device(args, format) {
                    Ok(opened) => return Ok(opened),
                    Err(err) => failures.push(format!("{}: {err}", format.label())),
                }
            }
            Err(format!(
                "failed to open {} with auto formats S32_LE/S24_LE/S24_3LE/S16_LE: {}",
                args.device,
                failures.join("; ")
            ))
        }
    }
}

fn wait_for_start_threshold(
    ring: &std::fs::File,
    args: &Args,
    mut header: RingHeader,
) -> Result<RingHeader, String> {
    if args.start_threshold_frames == 0 {
        return Ok(header);
    }
    let threshold = args.start_threshold_frames.min(header.capacity_frames);
    let started = Instant::now();
    while header.available_frames() < threshold {
        thread::sleep(Duration::from_millis(10));
        header = read_ring_header(ring)?;
        if started.elapsed() > Duration::from_secs(60) {
            return Err(format!(
                "timed out waiting for {} ring frames before playback; available={}",
                threshold,
                header.available_frames()
            ));
        }
    }
    Ok(header)
}

fn open_playback_device(args: &Args, format: PlaybackFormat) -> Result<PlaybackDevice, String> {
    let device = &args.device;
    let pcm = PCM::new(device, Direction::Playback, false)
        .map_err(|err| format!("failed to open ALSA playback {device}: {err}"))?;
    let negotiated_period_frames;
    let negotiated_buffer_frames;
    {
        let hwp =
            HwParams::any(&pcm).map_err(|err| format!("failed to read ALSA hw params: {err}"))?;
        hwp.set_format(alsa_format(format)).map_err(|err| {
            format!(
                "failed to set playback format {} on {device}: {err}",
                format.label()
            )
        })?;
        hwp.set_channels(DEFAULT_CHANNELS)
            .map_err(|err| format!("failed to set playback channels on {device}: {err}"))?;
        hwp.set_rate_resample(false).map_err(|err| {
            format!("failed to disable ALSA playback resampling on {device}: {err}")
        })?;
        hwp.set_rate(DEFAULT_SAMPLE_RATE, ValueOr::Nearest)
            .map_err(|err| {
                format!("failed to set playback rate {DEFAULT_SAMPLE_RATE} on {device}: {err}")
            })?;
        hwp.set_access(Access::RWInterleaved).map_err(|err| {
            format!("failed to set interleaved playback access on {device}: {err}")
        })?;
        hwp.set_period_size_near(args.period_frames as Frames, ValueOr::Nearest)
            .map_err(|err| {
                format!(
                    "failed to set ALSA period size near {} on {device}: {err}",
                    args.period_frames
                )
            })?;
        hwp.set_buffer_size_near((args.period_frames * args.alsa_buffer_periods) as Frames)
            .map_err(|err| {
                format!(
                    "failed to set ALSA buffer near {} frames on {device}: {err}",
                    args.period_frames * args.alsa_buffer_periods
                )
            })?;
        pcm.hw_params(&hwp)
            .map_err(|err| format!("failed to apply ALSA playback params on {device}: {err}"))?;

        let current = pcm
            .hw_params_current()
            .map_err(|err| format!("failed to read ALSA negotiated params on {device}: {err}"))?;
        let rate = current
            .get_rate()
            .map_err(|err| format!("failed to read negotiated rate on {device}: {err}"))?;
        let channels = current
            .get_channels()
            .map_err(|err| format!("failed to read negotiated channels on {device}: {err}"))?;
        let negotiated_format = current
            .get_format()
            .map_err(|err| format!("failed to read negotiated format on {device}: {err}"))?;
        if rate != DEFAULT_SAMPLE_RATE
            || channels != DEFAULT_CHANNELS
            || negotiated_format != alsa_format(format)
        {
            let _ = pcm.drop();
            return Err(format!(
                "ALSA returned rate={rate}, channels={channels}, format={negotiated_format}; expected {DEFAULT_SAMPLE_RATE}/2/{}",
                format.label()
            ));
        }
        negotiated_period_frames = current
            .get_period_size()
            .map_err(|err| format!("failed to read negotiated period on {device}: {err}"))?
            as usize;
        negotiated_buffer_frames = current
            .get_buffer_size()
            .map_err(|err| format!("failed to read negotiated buffer on {device}: {err}"))?
            as usize;
        if negotiated_period_frames == 0 || negotiated_buffer_frames == 0 {
            let _ = pcm.drop();
            return Err(format!(
                "ALSA returned period={negotiated_period_frames}, buffer={negotiated_buffer_frames}"
            ));
        }

        let swp = pcm
            .sw_params_current()
            .map_err(|err| format!("failed to read ALSA sw params on {device}: {err}"))?;
        swp.set_avail_min(negotiated_period_frames as Frames)
            .map_err(|err| format!("failed to set ALSA avail_min on {device}: {err}"))?;
        swp.set_start_threshold(negotiated_buffer_frames as Frames)
            .map_err(|err| format!("failed to set ALSA start threshold on {device}: {err}"))?;
        pcm.sw_params(&swp)
            .map_err(|err| format!("failed to apply ALSA sw params on {device}: {err}"))?;
        pcm.prepare()
            .map_err(|err| format!("failed to prepare ALSA playback on {device}: {err}"))?;
    }
    Ok(PlaybackDevice {
        pcm,
        format,
        period_frames: negotiated_period_frames,
        buffer_frames: negotiated_buffer_frames,
    })
}

fn alsa_format(format: PlaybackFormat) -> Format {
    match format {
        PlaybackFormat::S16Le => Format::s16(),
        PlaybackFormat::S24Le => Format::s24(),
        PlaybackFormat::S24_3Le => Format::s24_3(),
        PlaybackFormat::S32Le => Format::s32(),
    }
}

fn read_ring_header(ring: &std::fs::File) -> Result<RingHeader, String> {
    let mut bytes = [0u8; RING_HEADER_SIZE];
    ring.read_exact_at(&mut bytes, 0)
        .map_err(|err| format!("failed to read ring header: {err}"))?;
    RingHeader::parse(&bytes)
}

fn write_ring_header(ring: &std::fs::File, header: &RingHeader) -> Result<(), String> {
    ring.write_all_at(&header.to_bytes(), 0)
        .map_err(|err| format!("failed to write ring header: {err}"))
}

fn read_s16_period_or_zero_fill(
    ring: &std::fs::File,
    header: &RingHeader,
    period_frames: usize,
) -> Result<(Vec<i16>, u64, u64), String> {
    let available = header.available_frames() as usize;
    let readable = period_frames.min(available);
    let missing = period_frames - readable;
    let mut bytes = vec![0u8; period_frames * RING_BYTES_PER_FRAME];
    if readable > 0 {
        read_ring_frames(ring, header, header.read_cursor, readable, &mut bytes)?;
    }
    let samples = bytes
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes(chunk.try_into().expect("chunk size checked")))
        .collect();
    Ok((samples, readable as u64, missing as u64))
}

fn read_ring_frames(
    ring: &std::fs::File,
    header: &RingHeader,
    cursor: u64,
    frames: usize,
    output: &mut [u8],
) -> Result<(), String> {
    let start_frame = (cursor % header.capacity_frames) as usize;
    let first_frames = frames.min(header.capacity_frames as usize - start_frame);
    let first_bytes = first_frames * RING_BYTES_PER_FRAME;
    ring.read_exact_at(
        &mut output[..first_bytes],
        ring_data_offset_for_frame(header, cursor) as u64,
    )
    .map_err(|err| format!("failed to read ring audio frames: {err}"))?;
    let remaining_frames = frames - first_frames;
    if remaining_frames > 0 {
        let remaining_bytes = remaining_frames * RING_BYTES_PER_FRAME;
        ring.read_exact_at(
            &mut output[first_bytes..first_bytes + remaining_bytes],
            RING_HEADER_SIZE as u64,
        )
        .map_err(|err| format!("failed to read wrapped ring audio frames: {err}"))?;
    }
    Ok(())
}

fn write_period(
    pcm: &PCM,
    io: &alsa::pcm::IO<u8>,
    payload: &[u8],
    period_frames: usize,
    metrics: &mut HostMetrics,
) -> Result<(), String> {
    let mut offset = 0usize;
    let bytes_per_frame = payload.len() / period_frames.max(1);
    while offset < payload.len() {
        match io.writei(&payload[offset..]) {
            Ok(frames) if frames > 0 => offset = offset.saturating_add(frames * bytes_per_frame),
            Ok(_) => {}
            Err(err) if err.errno() == 11 => {
                let _ = pcm.wait(Some(20));
                thread::sleep(Duration::from_millis(1));
            }
            Err(err) => {
                metrics.alsa_xruns = metrics.alsa_xruns.saturating_add(1);
                if pcm.try_recover(err, true).is_ok() {
                    continue;
                }
                return Err(format!("ALSA write failed after recovery attempt: {err}"));
            }
        }
    }
    Ok(())
}

fn print_stats(args: &Args, playback: &PlaybackDevice, metrics: &HostMetrics, header: &RingHeader) {
    let elapsed = metrics.started.elapsed().as_secs_f64();
    let frames_per_second = if elapsed > 0.0 {
        metrics.played_frames as f64 / elapsed
    } else {
        0.0
    };
    println!(
        "{{\"schema\":\"mrt_alsa_host.stats.v1\",\"ring\":\"{}\",\"device\":\"{}\",\"format\":\"{}\",\"period_frames\":{},\"alsa_buffer_frames\":{},\"elapsed_seconds\":{:.3},\"played_frames\":{},\"frames_per_second\":{:.1},\"ring_available_frames\":{},\"ring_underrun_frames\":{},\"ring_overrun_frames\":{},\"alsa_xruns\":{},\"zero_fill_events\":{},\"low_water_frames\":{}}}",
        args.ring.display(),
        args.device,
        playback.format.label(),
        playback.period_frames,
        playback.buffer_frames,
        elapsed,
        metrics.played_frames,
        frames_per_second,
        header.available_frames(),
        header.underrun_frames,
        header.overrun_frames,
        metrics.alsa_xruns,
        metrics.zero_fill_events,
        metrics.low_water_frames,
    );
}
