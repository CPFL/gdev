#include <stdio.h>
#include <stdlib.h>

int cuda_test_mmul(unsigned int n, char *path);

int main(int argc, char *argv[])
{
	int rc;
	unsigned int n = 3;
	
	if (argc > 1)
		n = atoi(argv[1]);

	rc = cuda_test_mmul(n, ".");
	if ( rc != 0)
		printf("Test failed\n");
	else
		printf("Test passed\n");
	
	return rc;

}
