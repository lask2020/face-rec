#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Color definitions for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Face Recognition System Deployment Tool ===${NC}"

# Check if docker is installed and running
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: docker command not found. Please install Docker first.${NC}"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo -e "${RED}Error: Docker daemon is not running. Please start Docker first.${NC}"
    exit 1
fi

# Function to show usage/help
show_help() {
    echo "Usage: ./deploy.sh [command]"
    echo ""
    echo "Commands:"
    echo "  (no arguments)   Full redeploy (down, build, and start all services)"
    echo "  all              Full redeploy (same as above)"
    echo "  restart          Restart all services without rebuilding"
    echo "  down             Stop all services"
    echo "  ps / status      Show status of all services"
    echo "  logs [service]   View logs of all services or a specific service (e.g., ./deploy.sh logs ai-worker)"
    echo ""
    echo "Rebuild & restart specific services:"
    echo "  control-plane    Redeploy Go Control Plane"
    echo "  ingestion        Redeploy Go Ingestion Worker"
    echo "  ai-worker        Redeploy Python AI Worker"
    echo "  frontend         Redeploy React Frontend"
    echo "  go2rtc           Redeploy go2rtc Stream Hub"
    echo "  rustfs           Redeploy RustFS Storage"
    echo "  qdrant           Redeploy Qdrant Vector DB"
    echo "  postgres         Redeploy PostgreSQL"
    echo "  redis            Redeploy Redis"
    echo ""
}

# Check argument
CMD=${1:-"all"}

case "$CMD" in
    help|--help|-h)
        show_help
        exit 0
        ;;
    all)
        echo -e "${YELLOW}Performing full redeploy...${NC}"
        echo -e "${YELLOW}Stopping all services...${NC}"
        docker compose down
        
        echo -e "${YELLOW}Building and starting all services in background...${NC}"
        docker compose up --build -d
        
        echo -e "${GREEN}Deploy complete! Checking status...${NC}"
        docker compose ps
        ;;
    restart)
        echo -e "${YELLOW}Restarting all services...${NC}"
        docker compose restart
        echo -e "${GREEN}Restart complete! Checking status...${NC}"
        docker compose ps
        ;;
    down)
        echo -e "${YELLOW}Stopping all services...${NC}"
        docker compose down
        echo -e "${GREEN}All services stopped.${NC}"
        ;;
    ps|status)
        echo -e "${BLUE}Current service status:${NC}"
        docker compose ps
        ;;
    logs)
        SERVICE=$2
        if [ -z "$SERVICE" ]; then
            echo -e "${BLUE}Showing logs for all services (Press Ctrl+C to exit)...${NC}"
            docker compose logs -f
        else
            echo -e "${BLUE}Showing logs for service '$SERVICE' (Press Ctrl+C to exit)...${NC}"
            docker compose logs -f "$SERVICE"
        fi
        ;;
    control-plane|ingestion|ai-worker|frontend|go2rtc|rustfs|qdrant|postgres|redis)
        echo -e "${YELLOW}Redeploying service: $CMD...${NC}"
        echo -e "${YELLOW}Stopping $CMD...${NC}"
        docker compose stop "$CMD"
        docker compose rm -f "$CMD"
        
        echo -e "${YELLOW}Rebuilding and starting $CMD...${NC}"
        docker compose up --build -d "$CMD"
        
        echo -e "${GREEN}Service $CMD redeployed successfully!${NC}"
        echo -e "${BLUE}Current status of $CMD:${NC}"
        docker compose ps "$CMD"
        ;;
    *)
        echo -e "${RED}Unknown command: $CMD${NC}"
        show_help
        exit 1
        ;;
esac
