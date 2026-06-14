// asr-server: OpenAI compatible HTTP server exposing /v1/audio/transcriptions
// over the public ABI. No SSL, meant to sit behind a reverse proxy. The
// transcription handler wires to qa_transcribe once the pipeline lands.

#include "httplib.h"
#include "qwenasr.h"
#include "utf8.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

static int main_impl(int argc, char **argv) {
  const char *host = "127.0.0.1";
  int port = 8090;
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--host") == 0 && i + 1 < argc) {
      host = argv[++i];
    } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
      port = atoi(argv[++i]);
    }
  }

  httplib::Server srv;
  srv.Post("/v1/audio/transcriptions", [](const httplib::Request &,
                                          httplib::Response &res) {
    res.status = 501;
    res.set_content("{\"error\":\"transcription not implemented yet\"}",
                    "application/json");
  });

  fprintf(stderr, "asr-server %s listening on %s:%d\n", qa_version(), host,
          port);
  srv.listen(host, port);
  return 0;
}

int main(int argc, char **argv) {
  utf8_init(&argc, &argv);
  try {
    return main_impl(argc, argv);
  } catch (const std::exception &e) {
    fprintf(stderr, "[asr-server] FATAL: %s\n", e.what());
    return 1;
  }
}
