```
helm --kubeconfig=kubeconfig.yaml upgrade -f predictor-custom.yaml \
    --create-namespace --namespace mlsd-predictor \
    --install predictor ml-system-design-22--predictor-1.0.0.tgz
```
