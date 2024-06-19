def run_script(module_name):
    module = __import__(module_name)
    module.main()

if __name__ == "__main__":
    scripts = ["MyModel70", "MyModel80", "MyModel90"]

    for script in scripts:
        print(f"Running {script}...")
        run_script(script)