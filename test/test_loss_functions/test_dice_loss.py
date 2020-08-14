import unittest
import torch
from loss_functions import dice_loss as dice


class TestDiceLoss(unittest.TestCase):

    def test_mean_tensor(self):
        y = torch.FloatTensor([1, 1, 1])
        self.assertEqual(dice.mean_tensor(y, 0), torch.FloatTensor([1]))

    def test_softmax_helper(self):
        input = torch.FloatTensor([[1,1,1], [2,2,2]])

if __name__ == '__main__':
    unittest.main()