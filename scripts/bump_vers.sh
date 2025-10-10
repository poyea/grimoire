#!/bin/bash

# Version bump script for grimoire
# Usage: ./scripts/bump_vers.sh [major|minor|patch]
# If no argument is provided, defaults to patch bump

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

usage() {
    echo "Usage: $0 [major|minor|patch]"
    echo ""
    echo "Bumps the version and creates a new git tag."
    echo ""
    echo "Arguments:"
    echo "  major  - Bump major version (X.0.0)"
    echo "  minor  - Bump minor version (x.Y.0)"
    echo "  patch  - Bump patch version (x.y.Z) [default]"
    echo ""
    echo "Examples:"
    echo "  $0 major   # 1.2.3 -> 2.0.0"
    echo "  $0 minor   # 1.2.3 -> 1.3.0"
    echo "  $0 patch   # 1.2.3 -> 1.2.4"
    exit 1
}

if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo -e "${RED}Error: Not a git repository${NC}"
    exit 1
fi

if ! git diff-index --quiet HEAD --; then
    echo -e "${YELLOW}Warning: You have uncommitted changes${NC}"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

BUMP_TYPE=${1:-patch}

if [[ ! "$BUMP_TYPE" =~ ^(major|minor|patch)$ ]]; then
    echo -e "${RED}Error: Invalid bump type '$BUMP_TYPE'${NC}"
    usage
fi

LATEST_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "v0.0.0")
echo -e "Current version: ${GREEN}$LATEST_TAG${NC}"

VERSION=${LATEST_TAG#v}

IFS='.' read -r -a VERSION_PARTS <<< "$VERSION"
MAJOR=${VERSION_PARTS[0]:-0}
MINOR=${VERSION_PARTS[1]:-0}
PATCH=${VERSION_PARTS[2]:-0}

case "$BUMP_TYPE" in
    major)
        MAJOR=$((MAJOR + 1))
        MINOR=0
        PATCH=0
        ;;
    minor)
        MINOR=$((MINOR + 1))
        PATCH=0
        ;;
    patch)
        PATCH=$((PATCH + 1))
        ;;
esac

NEW_VERSION="v${MAJOR}.${MINOR}.${PATCH}"
echo -e "New version: ${GREEN}$NEW_VERSION${NC}"

read -p "Create tag $NEW_VERSION? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

git tag -a "$NEW_VERSION" -m "Release $NEW_VERSION"
echo -e "${GREEN}✓${NC} Tag $NEW_VERSION created"

read -p "Push tag to remote? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    git push origin "$NEW_VERSION"
    echo -e "${GREEN}✓${NC} Tag pushed to remote"
    echo -e "\n${GREEN}Release workflow will now build and publish PDFs${NC}"
else
    echo -e "\n${YELLOW}Tag created locally. Push later with:${NC}"
    echo "  git push origin $NEW_VERSION"
fi
