#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>

static int exec_with_program_argv0(const char *program, int argc, char **argv) {
    // Stack argv (no malloc => no corruption)
    char *new_argv[argc + 1];
    new_argv[0] = (char *)program;
    for (int i = 1; i < argc; i++) {
        new_argv[i] = argv[i];
    }
    new_argv[argc] = NULL;

    execv(program, new_argv);

    // If execv returns, it's an error
    perror(program);
    return 1;
}

int main(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-ivfpq") == 0) {
            return exec_with_program_argv0("./IVFPQ/search", argc, argv);
        } else if (strcmp(argv[i], "-ivfflat") == 0) {
            return exec_with_program_argv0("./IVFFlat/search", argc, argv);
        } else if (strcmp(argv[i], "-lsh") == 0) {
            return exec_with_program_argv0("./LSH_Project/lsh_app", argc, argv);
        } else if (strcmp(argv[i], "-hypercube") == 0) {
            return exec_with_program_argv0("./HYPERCUBE_Project/hypercube", argc, argv);
        }
    }

    fprintf(stderr,
            "Usage: %s [options]\n"
            "You must include one of the following flags:\n"
            "  -ivfpq       Run IVFPQ search\n"
            "  -ivfflat     Run IVFFlat search\n"
            "  -lsh         Run LSH search\n"
            "  -hypercube   Run Hypercube search\n",
            argv[0]);

    return 1;
}
