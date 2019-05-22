# Evaluation Suite

### Metrics

All metrics can be used either by passing test and reference segmentations as
parameters or by passing a `ConfusionMatrix` object. The latter is useful when many
metrics need to be computed, because the relevant computations are only done once.
All metrics assume binary segmentation inputs.

`ConfusionMatrix` has two important methods: `.get_matrix()`, which returns 4 ints for true positives, false positives, true negatives and false negatives, and
`.get_existence()`, which returns 4 bools, indicating whether test and reference
segmentations are all ones or all zeros. The latter is used when you specify
`nan_for_nonexisting=True` in metric calls to return NaN instead of 0 when the result
is undefined, i.e. would require dividing by 0.

### Evaluator

The `Evaluator` is a class that holds one test and one reference segmentation at a time that can contain multiple labels (one-hot encoding is not supported). It also holds a labels attribute than can either be a list of ints (or tuples of ints) or a dictionary
that maps from ints (or tuples of ints) to label names. A typical labels dictionary
could look like this:

```python
labels = {
    1: "Edema",
    2: "Enhancing Tumor",
    3: "Necrosis",
    (1, 2, 3): "Whole Tumor"
}
```

Labels in a tuple will be joined. If no labels are set, they will be automatically constructed from the unique entries in the segmentations upon evaluation. The Evaluator has both a regular set of metrics
that will always be computed and a set of advanced metrics that will only be computed
if `.evaluate(advanced=True)` is passed. The `.evaluate()` method is designed to
look for metric definitions in the current frame, so when you work in an interactive shell and redefine something there (e.g. for testing purposes), the newly defined metric will be used. You can also pass test and reference segmentations directly to evaluate to save calls to `.set_test()` and `.set_reference()`. `.evaluate()` will return a result dictionary and also save it in the `.result` attribute, so you can call `.to_array()` (numpy) or `.to_pandas()` (pandas) later. The resulting shape will be (labels x metrics). `.evaluate()` also takes additional `**metric_kwargs` that will be passed to each metric call.

### NiftiEvaluator

`NiftiEvaluator` redefines the `.set_test()` and `.set_reference()` methods of the `Evaluator` to take path strings instead of arrays. It will read the NIfTI files using SimpleITK, save the SimpleITK images in the `.test_nifti` and `.reference_nifti` attributes and set the arrays as test and reference segmentations. `.evaluate()` has an additional parameter `voxel_spacing`, which should be an iterable of floats. If the parameter is None, the spacing will be automatically read from the SimpleITK images. If you manually read the spacing from SimpleITK images, note that you have to reverse the ordering, because SimpleITK will return (z,y,x) ordering while we expect (x,y,z).

### Evaluating multiple segmentations

If you want to evaluate multiple test/reference pairs and get aggregate statistics, use the `aggregate_scores()` function. It expects an iterable of test/reference pairs and an evaluator (instance or type, will automatically initialize if necessary), which is the `NiftiEvaluator` by default. Test and reference will be set via `.set_test()` and `.set_reference()`, so make sure you're passing the right type for the evaluator. The method will return a dictionary that contains a list of all separate results as well as their mean:

```python
results = {
    "all": [
        {
            "Label": {
                "Metric": float,
                "Metric": float,
                ...
            },
            "Label": {
                ...
            },
            ...
        },
        {
            "Label": ...,
            "Label": ...,
            ...
        },
        ...
    ],
    "mean": {
        "Label": ...
        "Label": ...,
        ...
    }
}
```

`nanmean=True` will use `np.nanmean` instead of `np.mean`. It should be easy to adjust the code to compute arbitrary statistics, but at the moment only mean is supported. If you specify a `json_output_file`, a json file will be written that contains the result dictionary as well as additional information you can specify using the other `json_*` parameters:

```python
json = {
    "name": json_name,  # experiment name, not yours
    "description": json_description,  # a longer description so you know what you did
    "timestamp": "YYYY-MM-DD hh:mm:ss.ffffff",  # automatically generated
    "task": json_task  # the decathlon task
    "author": json_author  # probably Fabian :)
    "results": ...  # the above dictionary
    "id": 001122334455  # hash of other entries as unique id
}
```

`labels` is passed to the evaluator and `**metric_kwargs` is passed to all `.evaluate()` calls.