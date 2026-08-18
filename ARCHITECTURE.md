                         USER
                          │
                          ▼
                  ┌─────────────────┐
                  │  sksurv public  │
                  │      API        │
                  └────────┬────────┘
                           │
          ┌────────────────┼─────────────────┐
          │                │                 │
          ▼                ▼                 ▼
     Data handling      Models           Metrics
          │                │                 │
          ▼                ▼                 ▼
     sksurv.datasets   linear_model      sksurv.metrics
     sksurv.io         ensemble          sksurv.compare
     sksurv.util       tree              sksurv.nonparametric
                        svm
                        meta
          │                │                 │
          └────────────────┼─────────────────┘
                           │
                           ▼
                   scikit-learn
                           │
                           ▼
                NumPy / SciPy / Cython