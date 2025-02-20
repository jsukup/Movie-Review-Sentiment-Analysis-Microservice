import asyncio
from tortoise import Tortoise
from app.main import TORTOISE_ORM
import subprocess
import os


async def init_tortoise():
    print("Initializing Tortoise ORM...")
    await Tortoise.init(config=TORTOISE_ORM)
    print("Generating schemas...")
    await Tortoise.generate_schemas()
    print("Closing connections...")
    await Tortoise.close_connections()


def init_db():
    # Clean up any existing migration files
    if os.path.exists("migrations"):
        print("Removing existing migrations directory...")
        import shutil

        shutil.rmtree("migrations")
    if os.path.exists("aerich.ini"):
        print("Removing existing aerich.ini...")
        os.remove("aerich.ini")

    print("Running Tortoise initialization...")
    asyncio.run(init_tortoise())

    print("Running Aerich initialization...")
    try:
        # Initialize Aerich with the correct config
        subprocess.run(["aerich", "init", "-t", "app.main.TORTOISE_ORM"], check=True)

        print("Creating initial migration...")
        # Initialize the database
        subprocess.run(["aerich", "init-db"], check=True)

        print("Database initialization complete!")
    except subprocess.CalledProcessError as e:
        print(f"Error during database initialization: {e}")
        raise


if __name__ == "__main__":
    init_db()
