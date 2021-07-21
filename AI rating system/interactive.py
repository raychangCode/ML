"""
File: interactive.py
Name: Jui-Ting Chang
------------------------
This file uses the function interactivePrompt
from util.py to predict the reviews input by 
users on Console. Remember to read the weights
and build a Dict[str: float]
"""

from submission import *
from util import *
from collections import defaultdict

FILENAME = 'weights'


def main():
	d = defaultdict(float)
	with open(FILENAME, 'r', encoding='utf-8') as f:
		for line in f:
			d[line.split()[0]] = float(line.split()[1])

	feature_vector = extractWordFeatures
	interactivePrompt(feature_vector, d)


if __name__ == '__main__':
	main()