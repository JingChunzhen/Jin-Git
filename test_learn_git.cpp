#include<stdio.h>
#include<iostream.h>
string function()
{
	return "this is a modification\n";
}

int main()
{
	printf("hello world");
	printf(function);
	return 0;
}