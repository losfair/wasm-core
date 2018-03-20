use std::io::{Read, Write};

pub struct StreamManager {
    streams: Vec<Option<Box<Stream>>>,
    idx_pool: Vec<usize>
}

impl StreamManager {
    pub fn new() -> StreamManager {
        let mut mgr = StreamManager {
            streams: Vec::new(),
            idx_pool: Vec::new()
        };

        mgr.init_default();

        mgr
    }

    pub fn add_stream<S: Stream + 'static>(&mut self, s: S) -> usize {
        let id = match self.idx_pool.pop() {
            Some(v) => v,
            None => {
                self.streams.push(None);
                self.streams.len() - 1
            }
        };
        assert!(self.streams[id].is_none());
        self.streams[id] = Some(Box::new(s));
        id
    }

    pub fn remove_stream(&mut self, idx: usize) -> Option<Box<Stream>> {
        if idx >= self.streams.len() {
            return None;
        }

        let s = match self.streams[idx].take() {
            Some(v) => v,
            None => return None
        };
        self.idx_pool.push(idx);
        Some(s)
    }

    fn init_default(&mut self) {
        assert_eq!(self.add_stream(StdinStream {}), 0);
        assert_eq!(self.add_stream(StdoutStream {}), 1);
        assert_eq!(self.add_stream(StderrStream {}), 2);
    }

    pub fn read_stream(&mut self, fd: u32, data: &mut [u8]) -> i32 {
        let fd = fd as usize;

        if fd >= self.streams.len() {
            return -1;
        }

        let s = match self.streams[fd] {
            Some(ref mut v) => v,
            None => return -1
        };

        s.read(data)
    }

    pub fn write_stream(&mut self, fd: u32, data: &[u8]) -> i32 {
        let fd = fd as usize;

        if fd >= self.streams.len() {
            return -1;
        }

        let s = match self.streams[fd] {
            Some(ref mut v) => v,
            None => return -1
        };

        s.write(data)
    }
}

pub trait Stream {
    fn read(&mut self, data: &mut [u8]) -> i32 {
        -1
    }

    fn write(&mut self, data: &[u8]) -> i32 {
        -1
    }
}

pub struct StdinStream {

}

impl Stream for StdinStream {
    fn read(&mut self, data: &mut [u8]) -> i32 {
        match ::std::io::stdin().read(data) {
            Ok(n) => n as i32,
            Err(_) => -1
        }
    }
}

pub struct StdoutStream {

}

impl Stream for StdoutStream {
    fn write(&mut self, data: &[u8]) -> i32 {
        match ::std::io::stdout().write(data) {
            Ok(n) => n as i32,
            Err(_) => -1
        }
    }
}

pub struct StderrStream {

}

impl Stream for StderrStream {
    fn write(&mut self, data: &[u8]) -> i32 {
        match ::std::io::stderr().write(data) {
            Ok(n) => n as i32,
            Err(_) => -1
        }
    }
}
