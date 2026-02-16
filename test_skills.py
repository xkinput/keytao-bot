#!/usr/bin/env python3
"""
Test Skills System
测试 skills 系统加载和调用
"""
import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from keytao_bot.skills import SkillsManager


async def test_skills_loading():
    """Test loading skills"""
    print("=" * 60)
    print("Testing Skills System")
    print("=" * 60)
    
    # Create skills manager
    manager = SkillsManager()
    
    # Load all skills
    print("\n1️⃣ Loading skills...")
    manager.load_all_skills()
    
    # Check loaded tools
    tools = manager.get_tools()
    print(f"\n✅ Loaded {len(tools)} tools")
    
    for i, tool in enumerate(tools, 1):
        func = tool.get("function", {})
        print(f"   {i}. {func.get('name', 'Unknown')} - {func.get('description', 'No description')}")
    
    # Test keytao lookup by code
    print("\n2️⃣ Testing keytao_lookup_by_code...")
    lookup_func = manager.get_tool_function("keytao_lookup_by_code")
    
    if lookup_func:
        result = await lookup_func(code="nau")
        print(f"   Query result for 'nau':")
        if result.get("success"):
            for phrase in result.get("phrases", [])[:3]:
                print(f"     • {phrase['word']} ({phrase['code']}) [权重: {phrase['weight']}]")
        else:
            print(f"     ❌ Error: {result.get('error')}")
    else:
        print("   ❌ Function not found")
    
    # Test keytao lookup by word
    print("\n3️⃣ Testing keytao_lookup_by_word...")
    lookup_func = manager.get_tool_function("keytao_lookup_by_word")
    
    if lookup_func:
        result = await lookup_func(word="你好")
        print(f"   Query result for '你好':")
        if result.get("success"):
            for phrase in result.get("phrases", [])[:3]:
                print(f"     • {phrase['word']} → {phrase['code']} [权重: {phrase['weight']}]")
        else:
            print(f"     ❌ Error: {result.get('error')}")
    else:
        print("   ❌ Function not found")
    
    print("\n" + "=" * 60)
    print("✅ Skills system test completed")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_skills_loading())
