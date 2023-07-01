# OSML Model Runner

This package contains an application used to orchestrate the execution of ML models on large satellite images. The
application monitors an input queue for processing requests, decomposes the image into a set of smaller regions and
tiles, invokes an ML model endpoint with each tile, and finally aggregates all the results into a single output. The
application itself has been containerized and is designed to run on a distributed cluster of machines collaborating
across instances to process images as quickly as possible.

### Table of Contents
* [Getting Started](#getting-started)
  * [Key Design Concepts](#key-design-concepts)
    * [Image Tiling](#image-tiling)
    * [Lazy IO & Encoding Formats with Internal Tiles](#lazy-io--encoding-formats-with-internal-tiles)
    * [Tile Overlap and Merging Results](#tile-overlap-and-merging-results)
  * [Package Layout](#package-layout)
  * [Prerequisites](prerequisites)
  * [Development Environment](#development-environment)
  * [Running ModelRunner](#running-modelrunner)
  * [Infrastructure](#infrastructure)
    * [S3](#s3)
  * [Documentation](#documentation)
* [Support & Feedback](#support--feedback)
* [Security](#security)
* [License](#license)


## Getting Started

This application currently offers one build options:

1.) Build on an Amazon Linux 2 container (see provided Dockerfile):
* OS=amazonlinux:latest
* Python=3.10
* Miniconda=Miniconda3-latest-Linux-x86_64

### Key Design Concepts

#### Image Tiling

The images to be processed by this application are expected to range anywhere from 500MB to 500GB in size. The upper
bound is consistently growing as sensors become increasingly capable of collecting larger swaths of high resolution
data. To handle these images the application applies two levels of tiling. The first is region based tiling in which the
application breaks the full image up into pieces that are small enough for a single machine to handle. All regions after
the first are placed on a second queue so other model runners can start processing those regions in parallel. The second
tiling phase is to break each region up into individual chunks that will be sent to the ML models. Many ML model
containers are configured to process images that are between 512 and 2048 pixels in size so the full processing of a
large 200,000 x 200,000 satellite image can result in >10,000 requests.

#### Lazy IO & Encoding Formats with Internal Tiles

The images themselves are assumed to reside in S3 and are assumed to be compressed and encoded in such a way as to
facilitate piecewise access to tiles without downloading the entire image. The GDAL library, a frequently used open
source implementation of GIS data tools, has the ability to read images directly from S3 making use of partial range
reads to only download the part of the overall image necessary to process the region.

#### Tile Overlap and Merging Results

Many of the ML algorithms we expect to run will involve object detection or feature extraction. It is possible that
features of interest would fall on the tile boundaries and therefore be missed by the ML models because they are only
seeing a fractional object. This application mitigates that by allowing requests to specify an overlap region size that
should be tuned to the expected size of the objects. Each tile sent to the ML model will be cut from the full image
overlapping the previous by the specified amount. Then the results from each tile are aggregated with the aid of a
Non-Maximal Suppression algorithm used to eliminate duplicates in cases where an object in an overlap region was picked
up by multiple model runs.

### Package Layout

* **/src**: This is the Python implementation of this application.
* **/test**: Unit tests have been implemented using [pytest](https://docs.pytest.org).
* **/bin**: The entry point for the containerized application.
* **/scripts**: Utility scripts that are not part of the main application frequently used in development / testing.

### Prerequisites

First, ensure you have installed the following tools locally

- [docker](https://nodejs.org/en)
- [tox](https://tox.wiki/en/latest/installation.html)
- [osml cdk](https://github.com/aws-solutions-library-samples/osml-cdk-constructs) deployed into your aws account

### Development Environment

To run the container in a build/test mode and work inside it.

```shell
docker run -it -v `pwd`/:/home/ --entrypoint /bin/bash .
```

### Running ModelRunner

To start a job, place an ImageRequest on the ImageRequestQueue.

Sample ImageRequest:
```json
{
    "jobArn": "arn:aws:oversightml:<REGION>:<ACCOUNT>:ipj/<job_name>",
    "jobName": "<job_name>",
    "jobId": "<job_id>",
    "imageUrls": ["<image_url>"],
    "outputs": [
        {"type": "S3", "bucket": "<result_bucket_arn>", "prefix": "<job_name>/"},
        {"type": "Kinesis", "stream": "<result_stream_arn>", "batchSize": 1000}
    ],
    "imageProcessor": {"name": "<sagemaker_endpoint>", "type": "SM_ENDPOINT"},
    "imageProcessorTileSize": 2048,
    "imageProcessorTileOverlap": 50,
    "imageProcessorTileFormat": "< NITF | JPEG | PNG | GTIFF >",
    "imageProcessorTileCompression": "< NONE | JPEG | J2K | LZW >"
}
```

### Infrastructure

#### S3
When configuring S3 buckets for images and results, be sure to follow [S3 Security Best Practices](https://docs.aws.amazon.com/AmazonS3/latest/userguide/security-best-practices.html).

### Documentation

You can find documentation for this library in the `./doc` directory. Sphinx is used to construct a searchable HTML
version of the API documents.

```shell
tox -e docs
```

## Support & Feedback

To post feedback, submit feature ideas, or report bugs, please use the [Issues](https://github.com/aws-solutions-library-samples/osml-model-runner/issues) section of this GitHub repo.

If you are interested in contributing to OversightML Model Runner, see the [CONTRIBUTING](CONTRIBUTING.md) guide.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

MIT No Attribution Licensed. See [LICENSE](LICENSE).
