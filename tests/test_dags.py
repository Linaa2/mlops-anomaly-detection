import os
import sys

# Ensure project root is on path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Ensure dags folder is on path (for callbacks import)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "airflow", "dags"))


def test_build_failure_callback_returns_callable():
    from callbacks import build_failure_callback

    cb = build_failure_callback(email="test@example.com")
    assert callable(cb)


def test_data_pipeline_dag_loads():
    from airflow.models import DagBag

    dagbag = DagBag(dag_folder="airflow/dags", include_examples=False)
    assert "data_pipeline" in dagbag.dags, f"Import errors: {dagbag.import_errors}"
    assert len(dagbag.import_errors) == 0


def test_data_pipeline_task_ids():
    from airflow.models import DagBag

    dagbag = DagBag(dag_folder="airflow/dags", include_examples=False)
    dag = dagbag.dags["data_pipeline"]
    task_ids = {t.task_id for t in dag.tasks}
    expected = {
        "download_dataset",
        "unzip_dataset",
        "merge_sensors",
        "preprocess",
        "sample_data",
        "trigger_training",
    }
    assert task_ids == expected


def test_data_pipeline_task_order():
    from airflow.models import DagBag

    dagbag = DagBag(dag_folder="airflow/dags", include_examples=False)
    dag = dagbag.dags["data_pipeline"]

    # download -> unzip
    download = dag.get_task("download_dataset")
    assert "unzip_dataset" in {t.task_id for t in download.downstream_list}

    # preprocess -> sample_data -> trigger_training
    preprocess_task = dag.get_task("preprocess")
    assert "sample_data" in {t.task_id for t in preprocess_task.downstream_list}

    sample_task = dag.get_task("sample_data")
    assert "trigger_training" in {t.task_id for t in sample_task.downstream_list}


def test_data_pipeline_schedule_is_daily():
    from airflow.models import DagBag

    dagbag = DagBag(dag_folder="airflow/dags", include_examples=False)
    dag = dagbag.dags["data_pipeline"]
    # Airflow normalizes @daily to a timetable; check the original schedule
    assert dag.timetable.__class__.__name__ in ("CronTriggerTimetable", "CronDataIntervalTimetable")


def test_training_pipeline_dag_loads():
    from airflow.models import DagBag

    dagbag = DagBag(dag_folder="airflow/dags", include_examples=False)
    assert "training_pipeline" in dagbag.dags, f"Import errors: {dagbag.import_errors}"
    assert len(dagbag.import_errors) == 0


def test_training_pipeline_task_ids():
    from airflow.models import DagBag

    dagbag = DagBag(dag_folder="airflow/dags", include_examples=False)
    dag = dagbag.dags["training_pipeline"]
    task_ids = {t.task_id for t in dag.tasks}
    assert task_ids == {"train_model", "promote_or_reject"}


def test_training_pipeline_task_order():
    from airflow.models import DagBag

    dagbag = DagBag(dag_folder="airflow/dags", include_examples=False)
    dag = dagbag.dags["training_pipeline"]
    train_task = dag.get_task("train_model")
    downstream_ids = {t.task_id for t in train_task.downstream_list}
    assert "promote_or_reject" in downstream_ids


def test_training_pipeline_schedule_is_none():
    from airflow.models import DagBag

    dagbag = DagBag(dag_folder="airflow/dags", include_examples=False)
    dag = dagbag.dags["training_pipeline"]
    assert dag.timetable.__class__.__name__ == "NullTimetable"
