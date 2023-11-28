"""Deploy a number of agents in the moving peak experiment"""
import os
import sys
import pathlib
import numpy as np

from gwlandscape_python import GWLandscape

key_location = os.path.join(pathlib.Path.home(), 'vault/gwlandscape_key.txt')
api_key = str(np.genfromtxt(key_location, dtype='str'))

# Declare gwlandscape object
GWL = GWLandscape(token=api_key)

# Get or create the publication
PUB_DATA = {'author': 'Adam Boesky',
    'title': 'The Binary Black Hole Merger Rate Deviates from a Simple Delayed Cosmic Star Formation Rate: The Impact of Metallicity and Delay Time distributions',
    'arxiv_id': '000000',
    'year': 2023}
if len(GWL.get_publications(title=PUB_DATA['title'])) == 0:  # if it hasn't been made yet
    print(f'Creating publication with data:\n{PUB_DATA}')
    GWL.create_publication(**PUB_DATA)
elif len(GWL.get_publications(title=PUB_DATA['title'])) == 1:  # if it has been made
    print(f'Getting publication with title: {PUB_DATA["title"]}')
    PUB = GWL.get_publications(title=PUB_DATA['title'])[0]
else:  # if there are duplicate paper titles
    raise KeyError(f"There are {len(GWL.get_publications(title=PUB_DATA['title']))} publications with the given name!")


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
