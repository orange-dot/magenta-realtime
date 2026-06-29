use std::collections::BTreeMap;

pub const RING_MAGIC: &[u8; 8] = b"MRT2RNG1";
pub const RING_VERSION: u32 = 1;
pub const RING_HEADER_SIZE: usize = 128;
pub const RING_FORMAT_ID_S16_INTERLEAVED_STEREO: u32 = 1;
pub const DEFAULT_SAMPLE_RATE: u32 = 48_000;
pub const DEFAULT_CHANNELS: u32 = 2;
pub const DEFAULT_MODEL_FRAME_SIZE: u32 = 1_920;
pub const RING_BYTES_PER_FRAME: usize = DEFAULT_CHANNELS as usize * std::mem::size_of::<i16>();

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RingHeader {
    pub version: u32,
    pub header_size: u32,
    pub format_id: u32,
    pub sample_rate: u32,
    pub channels: u32,
    pub model_frame_size: u32,
    pub chunk_frames: u32,
    pub capacity_frames: u64,
    pub write_cursor: u64,
    pub read_cursor: u64,
    pub underrun_frames: u64,
    pub overrun_frames: u64,
    pub low_water_frames: u64,
}

impl RingHeader {
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < RING_HEADER_SIZE {
            return Err("ring header is too short".to_string());
        }
        if &bytes[0..8] != RING_MAGIC {
            return Err(format!("ring magic mismatch: {:?}", &bytes[0..8]));
        }
        let header = Self {
            version: read_u32(bytes, 8)?,
            header_size: read_u32(bytes, 12)?,
            format_id: read_u32(bytes, 16)?,
            sample_rate: read_u32(bytes, 20)?,
            channels: read_u32(bytes, 24)?,
            model_frame_size: read_u32(bytes, 28)?,
            chunk_frames: read_u32(bytes, 32)?,
            capacity_frames: read_u64(bytes, 36)?,
            write_cursor: read_u64(bytes, 44)?,
            read_cursor: read_u64(bytes, 52)?,
            underrun_frames: read_u64(bytes, 60)?,
            overrun_frames: read_u64(bytes, 68)?,
            low_water_frames: read_u64(bytes, 76)?,
        };
        header.validate()?;
        Ok(header)
    }

    pub fn to_bytes(&self) -> [u8; RING_HEADER_SIZE] {
        let mut bytes = [0u8; RING_HEADER_SIZE];
        bytes[0..8].copy_from_slice(RING_MAGIC);
        write_u32(&mut bytes, 8, self.version);
        write_u32(&mut bytes, 12, self.header_size);
        write_u32(&mut bytes, 16, self.format_id);
        write_u32(&mut bytes, 20, self.sample_rate);
        write_u32(&mut bytes, 24, self.channels);
        write_u32(&mut bytes, 28, self.model_frame_size);
        write_u32(&mut bytes, 32, self.chunk_frames);
        write_u64(&mut bytes, 36, self.capacity_frames);
        write_u64(&mut bytes, 44, self.write_cursor);
        write_u64(&mut bytes, 52, self.read_cursor);
        write_u64(&mut bytes, 60, self.underrun_frames);
        write_u64(&mut bytes, 68, self.overrun_frames);
        write_u64(&mut bytes, 76, self.low_water_frames);
        bytes
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.version != RING_VERSION {
            return Err(format!(
                "ring version mismatch: {} != {}",
                self.version, RING_VERSION
            ));
        }
        if self.header_size as usize != RING_HEADER_SIZE {
            return Err(format!(
                "ring header-size mismatch: {} != {}",
                self.header_size, RING_HEADER_SIZE
            ));
        }
        if self.format_id != RING_FORMAT_ID_S16_INTERLEAVED_STEREO {
            return Err(format!("unsupported ring format id: {}", self.format_id));
        }
        if self.sample_rate != DEFAULT_SAMPLE_RATE {
            return Err(format!(
                "ring sample-rate mismatch: {} != {}",
                self.sample_rate, DEFAULT_SAMPLE_RATE
            ));
        }
        if self.channels != DEFAULT_CHANNELS {
            return Err(format!(
                "ring channel mismatch: {} != {}",
                self.channels, DEFAULT_CHANNELS
            ));
        }
        if self.model_frame_size != DEFAULT_MODEL_FRAME_SIZE {
            return Err(format!(
                "ring model-frame mismatch: {} != {}",
                self.model_frame_size, DEFAULT_MODEL_FRAME_SIZE
            ));
        }
        if self.capacity_frames == 0 {
            return Err("ring capacity_frames must be positive".to_string());
        }
        if self.write_cursor < self.read_cursor {
            return Err("ring write cursor is behind read cursor".to_string());
        }
        if self.write_cursor - self.read_cursor > self.capacity_frames {
            return Err("ring cursors exceed capacity".to_string());
        }
        Ok(())
    }

    pub fn available_frames(&self) -> u64 {
        self.write_cursor.saturating_sub(self.read_cursor)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaybackFormat {
    S16Le,
    S24Le,
    S24_3Le,
    S32Le,
}

impl PlaybackFormat {
    pub fn label(self) -> &'static str {
        match self {
            Self::S16Le => "S16_LE",
            Self::S24Le => "S24_LE",
            Self::S24_3Le => "S24_3LE",
            Self::S32Le => "S32_LE",
        }
    }

    pub fn bytes_per_sample(self) -> usize {
        match self {
            Self::S16Le => 2,
            Self::S24_3Le => 3,
            Self::S24Le | Self::S32Le => 4,
        }
    }

    pub fn bytes_per_frame(self) -> usize {
        self.bytes_per_sample() * DEFAULT_CHANNELS as usize
    }

    pub fn parse(value: &str) -> Result<Self, String> {
        match value.trim().to_ascii_lowercase().as_str() {
            "s16" | "s16_le" | "s16le" => Ok(Self::S16Le),
            "s24" | "s24_le" | "s24le" => Ok(Self::S24Le),
            "s24_3" | "s24_3le" | "s24_3_le" => Ok(Self::S24_3Le),
            "s32" | "s32_le" | "s32le" => Ok(Self::S32Le),
            other => Err(format!("unsupported playback format {other:?}")),
        }
    }
}

pub fn auto_playback_format_order() -> [PlaybackFormat; 4] {
    [
        PlaybackFormat::S32Le,
        PlaybackFormat::S24Le,
        PlaybackFormat::S24_3Le,
        PlaybackFormat::S16Le,
    ]
}

pub fn convert_s16_interleaved_to_playback_format(
    frames: &[i16],
    format: PlaybackFormat,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(frames.len() * format.bytes_per_sample());
    for sample in frames {
        match format {
            PlaybackFormat::S16Le => out.extend_from_slice(&sample.to_le_bytes()),
            PlaybackFormat::S32Le => {
                let sample = i32::from(*sample) << 16;
                out.extend_from_slice(&sample.to_le_bytes());
            }
            PlaybackFormat::S24Le => {
                let sample = i32::from(*sample) << 8;
                out.extend_from_slice(&sample.to_le_bytes());
            }
            PlaybackFormat::S24_3Le => {
                let sample = i32::from(*sample) << 8;
                let bytes = sample.to_le_bytes();
                out.extend_from_slice(&bytes[..3]);
            }
        }
    }
    out
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlsaHwDevice {
    pub raw: String,
    pub fields: BTreeMap<String, String>,
}

pub fn parse_alsa_hw_device(value: &str) -> Result<AlsaHwDevice, String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err("ALSA device must not be empty".to_string());
    }
    let lowered = trimmed.to_ascii_lowercase();
    if lowered == "default" || lowered == "pipewire" || lowered.starts_with("plug") {
        return Err(format!(
            "refusing non-hardware ALSA PCM alias {trimmed:?}; use hw:CARD=...,DEV=..."
        ));
    }
    let Some(rest) = trimmed.strip_prefix("hw:") else {
        return Err(format!(
            "ALSA device must use hw: directly, got {trimmed:?}"
        ));
    };
    let mut fields = BTreeMap::new();
    for part in rest.split(',') {
        let Some((key, value)) = part.split_once('=') else {
            return Err(format!("invalid hw field {part:?}; expected KEY=VALUE"));
        };
        if key.is_empty() || value.is_empty() {
            return Err(format!("invalid hw field {part:?}; key/value required"));
        }
        fields.insert(key.to_string(), value.to_string());
    }
    if !fields.contains_key("CARD") || !fields.contains_key("DEV") {
        return Err("ALSA hw device must include CARD and DEV".to_string());
    }
    Ok(AlsaHwDevice {
        raw: trimmed.to_string(),
        fields,
    })
}

pub fn ring_data_offset_for_frame(header: &RingHeader, cursor: u64) -> usize {
    let frame = (cursor % header.capacity_frames) as usize;
    RING_HEADER_SIZE + frame * RING_BYTES_PER_FRAME
}

pub fn ring_file_size(header: &RingHeader) -> usize {
    RING_HEADER_SIZE + header.capacity_frames as usize * RING_BYTES_PER_FRAME
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, String> {
    let end = offset + 4;
    let slice = bytes
        .get(offset..end)
        .ok_or_else(|| format!("missing u32 at header offset {offset}"))?;
    Ok(u32::from_le_bytes(
        slice.try_into().expect("slice length checked"),
    ))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, String> {
    let end = offset + 8;
    let slice = bytes
        .get(offset..end)
        .ok_or_else(|| format!("missing u64 at header offset {offset}"))?;
    Ok(u64::from_le_bytes(
        slice.try_into().expect("slice length checked"),
    ))
}

fn write_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn write_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_python_ring_header_layout() {
        let mut bytes = vec![0u8; RING_HEADER_SIZE];
        bytes[0..8].copy_from_slice(RING_MAGIC);
        write_u32(&mut bytes, 8, RING_VERSION);
        write_u32(&mut bytes, 12, RING_HEADER_SIZE as u32);
        write_u32(&mut bytes, 16, RING_FORMAT_ID_S16_INTERLEAVED_STEREO);
        write_u32(&mut bytes, 20, DEFAULT_SAMPLE_RATE);
        write_u32(&mut bytes, 24, DEFAULT_CHANNELS);
        write_u32(&mut bytes, 28, DEFAULT_MODEL_FRAME_SIZE);
        write_u32(&mut bytes, 32, 4);
        write_u64(&mut bytes, 36, 4096);
        write_u64(&mut bytes, 44, 300);
        write_u64(&mut bytes, 52, 100);
        write_u64(&mut bytes, 60, 2);
        write_u64(&mut bytes, 68, 3);
        write_u64(&mut bytes, 76, 88);

        let header = RingHeader::parse(&bytes).expect("header should parse");

        assert_eq!(header.chunk_frames, 4);
        assert_eq!(header.capacity_frames, 4096);
        assert_eq!(header.available_frames(), 200);
        assert_eq!(header.underrun_frames, 2);
        assert_eq!(header.overrun_frames, 3);
        assert_eq!(header.low_water_frames, 88);
    }

    #[test]
    fn ring_header_serializes_back_to_same_layout() {
        let header = RingHeader {
            version: RING_VERSION,
            header_size: RING_HEADER_SIZE as u32,
            format_id: RING_FORMAT_ID_S16_INTERLEAVED_STEREO,
            sample_rate: DEFAULT_SAMPLE_RATE,
            channels: DEFAULT_CHANNELS,
            model_frame_size: DEFAULT_MODEL_FRAME_SIZE,
            chunk_frames: 8,
            capacity_frames: 8192,
            write_cursor: 64,
            read_cursor: 32,
            underrun_frames: 4,
            overrun_frames: 5,
            low_water_frames: 16,
        };

        let parsed = RingHeader::parse(&header.to_bytes()).expect("serialized header should parse");

        assert_eq!(parsed, header);
    }

    #[test]
    fn rejects_non_hw_alsa_aliases() {
        assert!(parse_alsa_hw_device("default").is_err());
        assert!(parse_alsa_hw_device("pipewire").is_err());
        assert!(parse_alsa_hw_device("plughw:CARD=AG06AG03,DEV=0").is_err());
    }

    #[test]
    fn parses_ag03_hw_device() {
        let parsed =
            parse_alsa_hw_device("hw:CARD=AG06AG03,DEV=0").expect("AG03 hw device should parse");

        assert_eq!(parsed.fields["CARD"], "AG06AG03");
        assert_eq!(parsed.fields["DEV"], "0");
    }

    #[test]
    fn converts_s16_ring_samples_to_yamaha_24_bit_containers() {
        let samples = [0i16, 1, -1, 32767, -32768];

        let s32 = convert_s16_interleaved_to_playback_format(&samples, PlaybackFormat::S32Le);
        let words: Vec<i32> = s32
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(words, vec![0, 65536, -65536, 2147418112, -2147483648]);

        let s24 = convert_s16_interleaved_to_playback_format(&samples, PlaybackFormat::S24_3Le);
        assert_eq!(&s24[0..3], &[0, 0, 0]);
        assert_eq!(&s24[3..6], &[0, 1, 0]);
        assert_eq!(&s24[6..9], &[0, 255, 255]);
    }
}
