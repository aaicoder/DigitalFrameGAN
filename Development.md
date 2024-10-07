
```
docker build -t digitalframe-gan .

docker run -it --gpus "all" --ipc=host --ulimit memlock=-1 -v "$(pwd):/src" digitalframe-gan /bin/bash
```

```
cog build -t digitalframe-gan
```


## Rub web server
```base
docker run -it -p 5001:5001 --gpus all -v "$(pwd):/src" digitalframe-gan
```