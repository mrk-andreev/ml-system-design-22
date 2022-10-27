```
docker-compose up --build
docker exec v4_airflow_airflow_1 mkdir /root/airflow/dags
docker cp airflow_docker_example.py v4_airflow_airflow_1:/root/airflow/dags/airflow_docker_example.py
```

Open `localhost:8080`

Find username and password in logs:

```
airflow_1  | standalone | Airflow is ready
airflow_1  | standalone | Login with username: admin  password: <PASSWORD>
airflow_1  | standalone | Airflow Standalone is for development purposes only. Do not use this in production!
```

