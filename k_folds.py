import numpy as np

class KFolds():
    def __init__(self, image_filenames, n_folds=3, test_fraction=0.3, seed=42):
        # Count the number of images
        n_images = len(image_filenames)
        assert n_images > 0
        # Compute the initial number of images in the training set
        assert (test_fraction >= 0) and (test_fraction <= 1)
        n_train  = int((1 - test_fraction) * n_images)
        # Ensure that the number of images in the training set is divisible by k
        assert n_folds >= 1
        remainder = n_train % n_folds
        n_train = n_train - remainder
        assert n_train % n_folds == 0

        self.image_filenames = np.array(image_filenames)
        self.n_folds = n_folds
        
        # Compute the number of images per fold.
        # (Used for fetching a fold.)
        self.n_images_per_fold = n_train // n_folds

        # Compute a reproducible, random permutation of the image indexes
        r = np.random.RandomState(seed=seed)
        idx_perm = r.permutation(n_images)

        self.train_idxs = idx_perm[:n_train]
        self.test_idxs  = idx_perm[n_train:]

    def __call__(self):
        return {
            'train': self.get_training_set(),
            'test': self.get_test_set(),
            'fold': [self.get_fold(i) for i in range(self.n_folds)]
        }


    def get_training_set(self):
        return self.image_filenames[self.train_idxs]

    def get_test_set(self):
        return self.image_filenames[self.test_idxs]

    def get_fold(self, n):
        assert isinstance(n, int)
        assert (n >= 0) and (n < self.n_folds)

        start = n * self.n_images_per_fold
        end   = start + self.n_images_per_fold

        fold_eval_idxs = self.train_idxs[start:end]
        fold_train_idxs = np.concatenate(
            [
                self.train_idxs[:start],
                self.train_idxs[end:]
            ])

        return self.image_filenames[fold_train_idxs], \
               self.image_filenames[fold_eval_idxs]
