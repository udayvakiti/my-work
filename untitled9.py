# Python3 program for
# the above approach
import sys
from collections import defaultdict

# Function to find the maximum
# absolute difference between
# distinct elements in arr[]
def MaxAbsDiff(arr, n):

	# HashMap to store each element
	# with their occurrence in array
	map = defaultdict (int)

	# maxElement and minElement to
	# store maximum and minimum
	# distinct element in arr[]
	maxElement = -sys.maxsize - 1
	minElement = sys.maxsize

	# Traverse arr[] and update each
	# element frequency in HashMap
	for i in range (n):
		map[arr[i]] += 1

	# Traverse HashMap and check if
	# value of any key appears 1
	# then update maxElement and
	# minElement by that key
	for k in map:
		if (map[k] == 1):
			maxElement = max(maxElement, k)
			minElement = min(minElement, k)

	# Return absolute difference of
	# maxElement and minElement
	return abs(maxElement - minElement)

# Driver Code
if __name__ == "__main__":

	# Given array arr[]
	arr = [10, 6, 5, 8]
	n = len( arr)

	# Function Call
	print(MaxAbsDiff(arr, n))

# This code is contributed by Chitranayal
