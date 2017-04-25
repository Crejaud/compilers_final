#include <iostream>
#include <stdio.h>
#include <fstream>
#include <string>
#include <permutations.cu>
using namespace std;

int main(int argc, char** argv) {
  // remove output file
  remove("cuda_permutations.out");

  // open output file
  ofstream outputFile;
  outputFile.open("cuda_permutations.out", ios::app);

  string usage =
		"\tRequired command line arguments:\n\
			Word length: E.g., --word_length 10\n\
                        Block size: E.g., --bsize 512\n\
                        Block count: E.g., --bcount 192\n";

  try {
    int bsize = 0, bcount = 0;
    int word_length = 1;

    // GET INPUT PARAMETERS

    for (int iii = 1; iii < argc; ++iii) {
      if (!strcmp(argv[iii], "--word_length") && iii != argc - 1) {
        word_length = atoi(argv[iii+1]);
      }
      else if (!strcmp(argv[iii], "--bsize") && iii != argc - 1) {
        bsize = atoi(argv[iii+1]);
      }
      else if (!strcmp(argv[iii], "--bcount") && iii != argc - 1) {
        bcount = atoi(argv[iii+1]);
      }
    }

    if (word_length <= 0 || bsize <= 0 || bcount <= 0) {
      cerr << "Usage: " << usage;
      return 1;
    }

    find_all_permutations(bsize, bcount, word_length, outputFile);
  }

}
