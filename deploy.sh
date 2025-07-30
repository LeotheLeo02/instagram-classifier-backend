#!/bin/bash

# Set environment variables
export PROJECT="newera-93301"    
export REGION="us-west1"     
export REPO="instagram-api"      
export TAG=$(date +%Y%m%d-%H%M%S)
export IMAGE="$REGION-docker.pkg.dev/$PROJECT/$REPO/api:$TAG"

# Build and submit the image
gcloud builds submit --tag "$IMAGE"

# Update the scrape-job with the new image
gcloud run jobs update scrape-job --image="$IMAGE" --region="$REGION"

echo "Deployment completed successfully!"
echo "Image: $IMAGE" 