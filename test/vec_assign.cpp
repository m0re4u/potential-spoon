// vector assignment
#include <iostream>
#include <vector>

int main ()
{
  std::vector<int> foo[2];
  std::vector<int>* bar[2];

  bar = &foo;
  std::cout << "Size of foo: " << int(foo.size()) << '\n';
  std::cout << "Size of bar: " << int(bar->size()) << '\n';
  foo.push_back(5);

  std::cout << "Size of foo: " << int(foo.size()) << '\n';
  std::cout << "Size of bar: " << int(bar->size()) << '\n';
  return 0;
}
