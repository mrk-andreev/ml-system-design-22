```
helm --kubeconfig=kubeconfig.yaml upgrade -f receiver-custom.yaml \
    --create-namespace --namespace mlsd-receiver \
    --install receiver ml-system-design-22--receiver-1.0.0.tgz
```
