import jwt


encoded_jwt = jwt.encode({'username': 'rajat'}, 'secret', algorithm='HS256')

print(encoded_jwt)

#https://localhost:3000/?jwt_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VybmFtZSI6InJhamF0In0.qNqNv6CDw_KMRV5gqDD-thJtvfqfJ6D-Ut-VFfYR490#/experiments/0?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D

#https://localhost:3000/?jwt_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VybmFtZSI6InJhamF0In0.qNqNv6CDw_KMRV5gqDD-thJtvfqfJ6D-Ut-VFfYR490#/experiments/0/
#https://localhost:3000/?jwt_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VybmFtZSI6InJhamF0In0.qNqNv6CDw_KMRV5gqDD-thJtvfqfJ6D-Ut-VFfYR490#/experiments/716008834097469325
