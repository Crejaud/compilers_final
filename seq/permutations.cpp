#include <iostream>
#include <vector>
#include <string>
#include <stdio.h>
#include <string.h>
#include <ctime>
#include <thread>
#include <mutex>
#include <fstream>
using namespace std;

ofstream out_rec, out_iter;

void find_permutations_rec(string);
void find_permutations_rec_helper(string, string);
void find_permutations(string, long long);

string setupWord(int);

static const char capital_letters[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

int main() {
  // remove output files
  remove("sequential_recursive.out");
  remove("sequential_iterative.out");

  // open output files
  out_rec.open("sequential_recursive.out", ios::app);
  out_iter.open("sequential_iterative.out", ios::app);

  clock_t start, end;
  double duration;
  int word_length, num_threads;
  string word = "";
  cout << "Please enter the integer size of your word: ";
  cin >> word_length;

  // create word that is nice and consistent for this problem
  word = setupWord(word_length);

  start = clock();
  // do sequential recursive
  find_permutations_rec(word);
  end = clock();

  duration = (end - start)/(double) CLOCKS_PER_SEC;

  cout << "[Sequential - Recursive] Permutations: " << duration * 1000 << " ms" << endl;

  start = clock();
  long long perm=1, digits=word.size();
  for (int k=1;k<=digits;perm*=k++);
  find_permutations(word, perm);

  end = clock();

  duration = (end - start)/(double) CLOCKS_PER_SEC;

  cout << "[Sequential - Iterative] Permutations: " << duration * 1000 << " ms" << endl;

  out_rec.close();
  out_iter.close();

  return 0;
}

void find_permutations(string word, long long num_perm) {
 for (long long first = 0;first < num_perm; first++) {
    string temp = word;

    long long div = num_perm;
    string perm = "";
    for (int digit = word.size(); digit > 0; digit--)
    {
      // compute the number of repetitions for one character in the actual column
      div /= digit;
      //compute the index of the character in the string
      long long t = first / div;
      int index = t % digit;
      perm += temp[index];
      //remove the used character
      temp.erase(index,1) ;
    }
    // append to file
    out_iter << perm << '\n';
  }
}

// create word of that length
string setupWord(int word_length) {
  string word;
  int rand_num;

  if (word_length <= 0) {
    cout << "Invalid size. Defaulting to size 10." << endl;
    word_length = 10;
  }

  for (int i = 0; i < word_length; i++) {
    rand_num = rand() % (sizeof(capital_letters) - 1);
    word += capital_letters[rand_num];
  }

  return word;
}

void find_permutations_rec(string word) {
  find_permutations_rec_helper("", word);
}

void find_permutations_rec_helper(string pre, string post) {
  if (post.empty()) {
    // found palindrome since pre is a palindrome
    out_rec << pre << '\n';
    return;
  }
  for (int i = 0; i < post.size(); i++) {
    find_permutations_rec_helper(pre + post[i],
      post.substr(0, i) + post.substr(i+1));
  }
}
