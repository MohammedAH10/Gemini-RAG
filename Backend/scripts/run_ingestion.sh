#!/bin/bash

###############################################################################
# Bulk Document Ingestion Runner
# Easy-to-use wrapper for bulk document ingestion into ChromaDB
###############################################################################

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print header
echo -e "${BLUE}================================================================${NC}"
echo -e "${GREEN}BULK DOCUMENT INGESTION - ChromaDB${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""

# Check if we're in the right directory
if [ ! -f "requirements.txt" ] || [ ! -d "app" ]; then
    echo -e "${RED}Error: Please run this script from the project root directory${NC}"
    exit 1
fi

# Function to display menu
show_menu() {
    echo -e "${YELLOW}Select an option:${NC}"
    echo "1) Test setup (verify prerequisites)"
    echo "2) Dry run (see what will be processed)"
    echo "3) Run full ingestion"
    echo "4) Check ChromaDB status"
    echo "5) Reset collection and re-ingest (WARNING: Deletes all data!)"
    echo "6) View ingestion logs"
    echo "7) Exit"
    echo ""
}

# Main menu loop
while true; do
    show_menu
    read -p "Enter choice (1-7): " choice
    
    case $choice in
        1)
            echo -e "\n${BLUE}Running setup verification...${NC}"
            python scripts/test_ingest_setup.py
            ;;
        2)
            echo -e "\n${BLUE}Running dry run...${NC}"
            python scripts/bulk_ingest.py --dry-run
            ;;
        3)
            echo -e "\n${YELLOW}Starting full document ingestion...${NC}"
            read -p "This will process 56 documents. Continue? (y/n): " confirm
            if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
                python scripts/bulk_ingest.py
            else
                echo "Cancelled."
            fi
            ;;
        4)
            echo -e "\n${BLUE}Checking ChromaDB status...${NC}"
            python scripts/check_chroma_status.py
            ;;
        5)
            echo -e "\n${RED}WARNING: This will delete all existing data in ChromaDB!${NC}"
            read -p "Are you sure? (y/n): " confirm
            if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
                python scripts/bulk_ingest.py --reset-collection
            else
                echo "Cancelled."
            fi
            ;;
        6)
            if [ -f "logs/bulk_ingest.log" ]; then
                echo -e "\n${BLUE}Recent ingestion logs:${NC}"
                tail -n 50 logs/bulk_ingest.log
            else
                echo -e "\n${YELLOW}No ingestion logs found yet.${NC}"
            fi
            ;;
        7)
            echo -e "\n${GREEN}Goodbye!${NC}"
            exit 0
            ;;
        *)
            echo -e "\n${RED}Invalid option. Please try again.${NC}\n"
            ;;
    esac
    
    echo ""
    read -p "Press Enter to continue..."
    clear
done
