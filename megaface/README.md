# MegaFace

Download Linux DevKit from [MagaFace WebSite](http://megaface.cs.washington.edu/) then extract:

```bash
$ tar -vxf linux-devkit.tar.gz
```

Delete previously generated feature files:
```bash
find facescrub_images -name "*.bin" -type f
find facescrub_images -name "*.bin" -type f -delete
find MegaFace/FlickrFinal2 -name "*.bin" -type f
find MegaFace/FlickrFinal2 -name "*.bin" -type f -delete
```
