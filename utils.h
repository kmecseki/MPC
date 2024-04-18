#pragma once
#ifndef MPC_UTILS_H
#define MPC_UTILS_H

#include <stdio.h>

void backup(FILE* fp);
void restore(const char* path);
void copyfile(const char* path, int dorestore);
FILE* opener(const char* cpath, const char* fname, const char* fopts, int dorestore);

#endif