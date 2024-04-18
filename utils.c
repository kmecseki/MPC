#include "utils.h"
#include <limits.h>

void backup(FILE* fp) {
    int fd;
    char* buffer = (char *)malloc(PATH_MAX);
    char* path = (char *)malloc(PATH_MAX);
    
    fd = fileno(fp);
    snprintf(buffer, PATH_MAX, "/proc/self/fd/%d", fd);
    memset(path, 0, PATH_MAX);
    readlink(buffer, path, PATH_MAX-1);
    
    free(buffer);
    
    copyfile(path, 0);
    
    free(path);
}


void restore(const char* path) {
    copyfile(path, 1);
}


void copyfile(const char* path, int dorestore) {

    char* buffer = (char *)malloc(2 * strlen(path) + 9);
    if (!buffer) {
        fprintf(stderr,"Memory allocation failed in copyfile function");
        free(buffer);
        exit(1);
    }
    if (dorestore) {
        sprintf(buffer, "cp %s.bak %s", path, path);
    } else {
        sprintf(buffer, "cp %s %s.bak", path, path);
    }
    system(buffer);
    if (system(buffer) == -1) {
        fprintf(stderr,"Error executing system command");
        free(buffer);
        exit(1);
    }
    free(buffer);
}


FILE* opener(const char* cpath, const char* fname, const char* fopts, int dorestore) {
    char* path;
    FILE* fp;
    
    // create pathname and open file
    path = (char *)malloc(strlen(cpath) + strlen(fname) + 2);
    if (!path) {
        fprintf(stderr, "Memory allocation failed in opener function\n");
        exit(1);
    }
    sprintf(path, "%s/%s", cpath, fname);
    if (dorestore) restore(path);
    fp = fopen(path, fopts);
    if (!fp) {
        fprintf(stderr, "Failed to open file: %s\n", path);
        exit(1);
    }
    free(path);
    return fp;
}