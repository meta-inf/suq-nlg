import sys, os
from google.cloud import aiplatform

sys.path.append(os.getcwd())

from generation_new.apis import get_gcp_project_id


print(sys.argv[1], sys.argv[2])
endpoint_id, location = sys.argv[2], sys.argv[1]
assert len(endpoint_id) > 5
project_id = get_gcp_project_id()
for endpoint in aiplatform.Endpoint.list(project=project_id, location=location):
    if endpoint.name.find(endpoint_id) != -1:
        endpoint.delete(force=True)
        sys.exit(0)

sys.exit(1)
