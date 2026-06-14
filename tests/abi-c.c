// abi-c.c: pure C99 smoke test locking in the public ABI. Compiles qwenasr.h
// with a C compiler under -Wall -Werror -pedantic and links the static lib.
// Never loads a model: failure means the public API regressed.

#include "qwenasr.h"

#include <stdio.h>

int main(void) {
    const char * v = qa_version();
    if (!v) {
        return 1;
    }

    struct qa_init_params ip = qa_init_default_params();
    if (ip.abi_version != QA_ABI_VERSION) {
        return 1;
    }

    struct qa_transcribe_params tp = qa_transcribe_default_params();
    if (tp.abi_version != QA_ABI_VERSION) {
        return 1;
    }

    printf("qwenasr abi ok %s\n", v);
    return 0;
}
