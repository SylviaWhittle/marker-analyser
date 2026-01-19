"""Code snippet from my helping Marina handle a dataset and obtain the plot data. (January 2026)"""

import matplotlib.pyplot as plt


def predict_and_plot_multi_datasets_single_fit_object(
    datasets,
    fit_object,
    model,
):
    """Predict and plot multiple datasets using a single fit object."""
    _fig, ax = plt.subplots(figsize=(10, 10))
    for label, (file_object, fd_key) in datasets.items():
        curve = file_object.fdcurves[fd_key]
        # Mask: restrict to desired range
        # todo: make configurable
        mask = (curve.f.data <= 20) & (curve.d.data > 0)
        _masked_forces = curve.f[mask].data
        masked_distances = curve.d[mask].data

        # model the forces
        kt_key = "kT"
        kt_value = fit_object.params[kt_key].value
        lc_key = "DNA/Lc_" + label
        lc_value = fit_object.params[lc_key].value
        lp_key = "DNA/Lp_" + label
        lp_value = fit_object.params[lp_key].value
        st_key = "DNA/St_" + label
        st_value = fit_object.params[st_key].value
        f_offset_key = "DNA/f_offset_" + label
        f_offset_value = fit_object.params[f_offset_key].value

        this_sample_params = {
            "DNA/Lc": lc_value,
            "DNA/Lp": lp_value,
            "DNA/St": st_value,
            "DNA/f_offset": f_offset_value,
            "kT": kt_value,
        }

        modelled_forces = model(independent=masked_distances, params=this_sample_params)
        ax.plot(masked_distances, modelled_forces, label=f"Model {label}")
    ax.set_title("After fitting")
    ax.set_xlabel(r"Distance [$\mu$m]")
    ax.legend(loc="upper right", bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.show()
