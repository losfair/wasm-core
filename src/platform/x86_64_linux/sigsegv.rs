use libc;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[allow(dead_code)]
pub struct Sigval {
    pub sival_ptr: *mut libc::c_void
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[allow(dead_code)]
pub struct SigsegvInfo {
    pub si_signo: libc::c_int,     /* Signal number */
    pub si_errno: libc::c_int,     /* An errno value */
    pub si_code: libc::c_int,      /* Signal code */
    pub __pad0: libc::c_int,
    pub si_addr: *mut libc::c_void
}
