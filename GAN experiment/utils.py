import numpy as np
import nibabel

def plot_central_cuts(img, title=""):
    """
    param image: tensor or np array of shape (CxDxHxW) if t is None
    """
    if isinstance(img, torch.Tensor):
        img = img.numpy()
        if (len(img.shape) > 3):
            img = img[0,:,:,:]
                
    elif isinstance(img, nibabel.nifti1.Nifti1Image):    
        img = img.get_fdata()
   
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(3 * 4, 4))
    axes[0].imshow(img[ img.shape[0] // 2, :, :], cmap='gray')
    axes[1].imshow(img[ :, img.shape[1] // 2, :], cmap='gray')
    axes[2].imshow(img[ :, :, img.shape[2] // 2], cmap='gray')
    axes[1].set_title(title)
    
    plt.show()
    
    
class Fake_MRIData(torch_data.Dataset):
    def __init__(self, X, y):
        super(Fake_MRIData, self).__init__()
        self.X = torch.tensor(X, dtype=torch.float32)[:, None, ...]
        self.y = torch.tensor(y, dtype=torch.float32)


    def __len__(self):

        return len(self.X)
    
    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]