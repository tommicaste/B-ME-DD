from real_data.utils.data import load_abide_data, load_adhd_data, load_frankfurt_data


# ------- Dataset Registry -------

DATASETS = {
    'abide': {
        'loader': load_abide_data,
        'loader_params': ['parcellation', 'roi_drop_frac'],
        'defaults': {'parcellation': 'rois_cc200', 'roi_drop_frac': 0.01},
        'valid_parcellations': ['rois_cc200', 'rois_cc400', 'rois_ho', 'rois_aal', 'yeo17', 'yeo7'],
        'subject_key': 'SUB_ID',
        'time_type': 'flexible',
    },
    'frankfurt': {
        'loader': load_frankfurt_data,
        'loader_params': ['parcellation'],
        'defaults': {'parcellation': 'large'},
        'valid_parcellations': ['large', 'small'],
        'subject_key': 'patient_id',
        'time_type': 'int',
    },
    'adhd': {
        'loader': load_adhd_data,
        'loader_params': ['parcellation', 'target'],
        'defaults': {'parcellation': 'cc200', 'target': 'inattentive'},
        'valid_parcellations': ['cc200', 'cc400', 'yeo17', 'yeo7'],
        'subject_key': 'subject_id',
        'time_type': 'flexible',
    },
}


# ------- Model Registry -------

MODELS = {
    'gnn': {
        'needs_adjacency': True,
        'needs_flatten': False,
        'is_deep': True,
        'variants': [('gcn', 1), ('gcn', 2), ('sage', 1), ('sage', 2)],
    },
    'mlp': {
        'needs_adjacency': False,
        'needs_flatten': True,
        'is_deep': True,
        'variants': None,
    },
    'lr': {
        'needs_adjacency': False,
        'needs_flatten': True,
        'is_deep': True,
        'variants': None,
    },
    'shallow': {
        'needs_adjacency': False,
        'needs_flatten': True,
        'is_deep': False,
        'variants': None,
    },
    'residualgnn': {
        'needs_adjacency': True,
        'needs_flatten': False,
        'is_deep': True,
        'variants': None,
    },
    'braingnn': {
        'needs_adjacency': True,
        'needs_flatten': False,
        'is_deep': True,
        'variants': None,
    },
}
