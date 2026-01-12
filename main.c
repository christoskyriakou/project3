#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>
#include <string.h>  

int main(int argc, char** argv) {

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-ivfpq") == 0) {
            char *program = "./IVFPQ/search";
            execv(program, argv);
        }
        else if (strcmp(argv[i], "-ivfflat") == 0) {
            char *program = "./IVFFlat/search";
            execv(program, argv);
        }
        else if (strcmp(argv[i], "-lsh") == 0) {
            char *program = "./LSH_Project/lsh_app";
            execv(program, argv);
        }
        else if (strcmp(argv[i], "-hypercube") == 0) {
            char *program = "./HYPERCUBE_Project/hypercube";
            execv(program, argv);
        }
    }

    fprintf(stderr,
        "Usage: %s [options]\n"
        "You must include one of the following flags:\n"
        "  -ivfpq       Run IVFPQ search\n"
        "  -ivfflat     Run IVFFlat search\n"
        "  -lsh         Run LSH search\n"
        "  -hypercube   Run Hypercube search\n",
        argv[0]
    );
    

    printf("wellp\n");

    return 0;
}