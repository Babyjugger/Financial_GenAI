from main import genAi_views
import asyncio

def main():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(genAi_views.main())


if __name__ == "__main__":
    main()
