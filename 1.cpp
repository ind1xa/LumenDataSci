#include <iostream>
#include <cstdlib>
using namespace std;

void bubble_sort(int *begin, int *end)
{
    int *i, *j;
    for (i = begin; i != end; i++)
    {
        for (j = begin; j != end - 1; j++)
        {
            if (*j > *(j + 1))
            {
                int temp = *j;
                *j = *(j + 1);
                *(j + 1) = temp;
            }
        }
    }
}

int main()
{
    // non sorted array
    int a[10000];

    for (int i = 0; i < 10000; i++)
    {
        a[i] = rand() % 10000;
    }

    // sorted array

    bubble_sort(a, a + 10000);

    for (int i = 0; i < 10000; i += 1000)
    {
        cout << a[i] << " ";
    }
    cout << endl;

    return 0;
}