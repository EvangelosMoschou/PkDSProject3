#include "io_utils.h"
#include <errno.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

int pread_full(int fd, void *buf, size_t count, off_t offset) {
  char *ptr = (char *)buf;
  size_t remaining = count;
  off_t current_offset = offset;
  int attempts = 0;
  const int MAX_ATTEMPTS = 1000;

  while (remaining > 0) {
    ssize_t read_bytes = pread(fd, ptr, remaining, current_offset);

    if (read_bytes < 0) {
      if (errno == EINTR) {
        continue;
      }
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        attempts++;
        if (attempts > MAX_ATTEMPTS) {
          fprintf(stderr, "pread_full: Too many EAGAIN failures\n");
          return -1;
        }
        // Exponential backoff
        struct timespec ts;
        ts.tv_sec = 0;
        ts.tv_nsec = (1000 * (1 << (attempts % 10))); // max ~1ms
        nanosleep(&ts, NULL);
        continue;
      }
      perror("pread_full failed");
      return -1;
    }

    if (read_bytes == 0) {
      fprintf(stderr, "pread_full: Unexpected EOF (wanted %zu more bytes)\n",
              remaining);
      return -1;
    }

    ptr += read_bytes;
    remaining -= read_bytes;
    current_offset += read_bytes;
    attempts = 0; // Reset attempts on successful read
  }

  return 0;
}
