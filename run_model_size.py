def run_script(module_name):
    module = __import__(module_name)
    module.main()

if __name__ == "__main__":
    scripts = ["MyModel70_1024"]
    for script in scripts:
        print(f"Running {script}...")
        run_script(script)