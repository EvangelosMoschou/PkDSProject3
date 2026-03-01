#ifndef IO_UTILS_H
#define IO_UTILS_H

#include <stddef.h>
#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

// Robustly read 'count' bytes from 'fd' at 'offset' into 'buf'.
// Handles EINTR, partial reads, and EAGAIN with backoff.
// Returns 0 on success, -1 on failure.
int pread_full(int fd, void *buf, size_t count, off_t offset);

#ifdef __cplusplus
}
#endif

#endif // IO_UTILS_H
