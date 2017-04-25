#include <vector>
#include <iostream>

#include <fstream>

__global__ void find_all_permutations_kernel(char* word, int word_length, unsigned long long num_perm, char* permutations) {
  unsigned long long thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned long long thread_num = blockDim.x * gridDim.x;

  unsigned long long warp_id = thread_id / 32;
  unsigned long long warp_num = thread_num % 32 == 0 ? thread_num / 32 : thread_num / 32 + 1;

  unsigned long long load = num_perm % warp_num == 0 ? num_perm / warp_num : num_perm / warp_num + 1;
  unsigned long long beg = load * warp_id;
  unsigned long long end = min(num_perm, beg + load);
  unsigned long long lane = thread_id % 32;
  beg += lane;

  for(int i = beg; i < end; i += 32) {

    char* temp = word;
    unsigned long long divisor = num_perm;
    //printf("divisor = %llu | num_perm = %llu | word_length %d\n", divisor, num_perm, word_length);
    unsigned long long permutations_index = 0;
    for (int digit = word_length; digit > 0; digit--) {
      //printf("divisor before = %llu, digit = %d\n", divisor, digit);
      divisor /= digit;
      //printf("divisor after = %llu\n", divisor);

      unsigned long long t = i / divisor;
      int index = t % digit;

      printf("permutations[%llu] = temp[%d] = %c | divisor = %llu | digit = %d | t = %llu\n", i*word_length + permutations_index, index, temp[index], divisor, digit, t);

      permutations[i*word_length + permutations_index] = temp[index];
      permutations_index++;

      // remove temp[index]
      memmove(&temp[index], &temp[index+1], strlen(temp) - index);
    }
  }
}

void generateWord(char* word, int* word_length);

char* find_all_permutations(int blockSize, int blockNum, int word_length) {
  // ALLOCATE
  char* word = (char *) malloc(word_length * sizeof(char));

  unsigned long long num_perm = 1;
  for (int k = 1; k <= word_length; num_perm *= k++);

  // generate word given length
  generateWord(word, &word_length);

  // this will contain all of the permutations of the word above
  char* permutations = (char *) malloc(word_length * num_perm * sizeof(char));

  char* cuda_permutations;
  char* cuda_word;

  cudaMalloc((void **) &cuda_permutations, word_length * num_perm * sizeof(char));
  cudaMalloc((void **) &cuda_word, word_length * sizeof(char));
  cudaMemcpy(cuda_permutations, permutations, word_length * num_perm * sizeof(char), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_word, word, word_length * sizeof(char), cudaMemcpyHostToDevice);

  // call kernel
  find_all_permutations_kernel<<<blockNum, blockSize>>>(cuda_word, word_length, num_perm, cuda_permutations);
  cudaDeviceSynchronize();

  cudaMemcpy(permutations, cuda_permutations, word_length * num_perm * sizeof(char), cudaMemcpyDeviceToHost);

  // DEALLOCATE
  cudaFree(cuda_permutations);
  cudaFree(cuda_word);
  free(word);

  return permutations;
}

/* Generate the random word given a word_length */
void generateWord(char* word, int* word_length) {
  int rand_num;

  const char capital_letters[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

  if (*word_length <= 0) {
    printf("Invalid size. Defaulting to size 10.\n");
    *word_length = 10;
  }

  for (int i = 0; i < *word_length; i++) {
    rand_num = rand() % (sizeof(capital_letters) - 1);
    word[i] = capital_letters[rand_num];
  }
}
