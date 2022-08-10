"""patch dataset."""

import tensorflow_datasets as tfds
import os
import glob
# TODO(patch): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(patch): BibTeX citation
_CITATION = """
"""


class Patch(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for patch dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(patch): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Image(shape=(None, None, 1)),
            'label': tfds.features.ClassLabel(names=['background', 'mass_ben', 'mass_mal']),
        }),

        supervised_keys=('image', 'label'),
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Download the data and define splits."""


    return {
        'train': self._generate_examples(path='/content/train'),
        'test': self._generate_examples(path='/content/test'),
        'val': self._generate_examples(path='/content/val'),

    }

  def _generate_examples(self, path):
    """Yields examples."""
    subfolders = [ f.path for f in os.scandir(path) if f.is_dir() ]

    for folder in subfolders:
      label = os.path.basename(folder)

      for image in os.listdir(folder):
        yield os.path.join(folder,image), {
            'image': os.path.join(folder,image),
            'label': label,
        }
