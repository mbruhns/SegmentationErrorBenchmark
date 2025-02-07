from scipy.sparse import csr_array, save_npz, load_npz
from skimage.segmentation import clear_border, relabel_sequential


def main():
    f1_val = 80

    mask_0_path = f"results/perturbed_masks/f1_{f1_val}/LHCC35_pert{f1_val}_0.npz"

    mask_1_path = f"results/perturbed_masks/f1_{f1_val}/LHCC35_pert{f1_val}_1.npz"

    pert_mask_0 = load_npz(mask_0_path).todense().astype("uint32", copy=False)
    pert_mask_1 = load_npz(mask_1_path).todense().astype("uint32", copy=False)

    mask_slice = slice(4000, 4500)

    pert_mask_0 = pert_mask_0[mask_slice, mask_slice].copy()
    pert_mask_1 = pert_mask_1[mask_slice, mask_slice].copy()

    print(pert_mask_0.sum())

    pert_mask_0, _, _ = relabel_sequential(clear_border(pert_mask_0))
    pert_mask_1, _, _ = relabel_sequential(clear_border(pert_mask_1))

    pert_mask_0 = csr_array(pert_mask_0)
    pert_mask_1 = csr_array(pert_mask_1)

    print(pert_mask_0.sum())

    save_npz("results/mini_masks/minimask0.npz", pert_mask_0)
    save_npz("results/mini_masks/minimask1.npz", pert_mask_1)

    foo = load_npz("results/mini_masks/minimask0.npz")
    print(foo.sum())


if __name__ == "__main__":
    main()
