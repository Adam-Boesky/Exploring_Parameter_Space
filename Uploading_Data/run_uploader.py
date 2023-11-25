"""Deploy a number of agents in the moving peak experiment"""
import os
import sys

from gwlandscape_python import GWLandscape

GWL = GWLandscape(token='1446a1b3c8be718461913e2f6397e5bda92bf4959b65dd284374381a7f89cedc')
PUB = GWL.create_publication(
    author='Adam Boesky',
    title='The Binary Black Hole Merger Rate Deviates from a Simple Delayed Cosmic Star Formation Rate: The Impact of Metallicity and Delay Time distributions',
    arxiv_id='000000',
    year=2023
)


def run_agent():
    fpath = sys.argv[-1]
    beta = sys.argv[-2]
    alpha = sys.argv[-3]

    print(f'Beginning a = {alpha}, b = {beta} upload!')
    model = GWL.create_model(name=f'alpha = {alpha}, beta = {beta}')
    dataset = GWL.create_dataset(
        publication=PUB,
        model=model,
        datafile=fpath
    )
    print('Dset: ', dataset)


if __name__=='__main__':
	run_agent()
