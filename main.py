import os
from dotenv import load_dotenv
from crew import CompetitiveAnalysisCrew

load_dotenv()


def run():
    print("Starting competitive analysis...")

    # 修改這段描述為你的產品資訊
    # user_input = "我們的產品是一個 AI 記帳 APP，用戶可以拍照記錄消費，AI 自動辨識金額與類別，並提供每月消費分析報告與預算建議。目標客群是 18-35 歲的年輕上班族，支援 iOS 和 Android。"
    # user_input = "kawasaki zx-25rr 是一款四缸小排量運動型摩托車，搭載 249cc 水冷四行程引擎，最大馬力約 50 匹，適合年輕騎士和初學者。它具有輕量化車架、優秀的操控性和現代化的設計，提供出色的性能和騎乘體驗。"
    # user_input = "Product:EasySep™ Mouse MDSC (CD11b+Gr1+) Isolation Kit. Description:Easily and efficiently isolate highly purified mouse myeloid-derived suppressor cells (MDSCs) (CD11b+Gr1+) from mouse splenocytes, bone marrow, or peripheral blood samples by immunomagnetic negative selection, with the EasySep™ Mouse MDSC (CD11b+Gr1+) Isolation Kit."
    # user_input = "產品：全自動化社群行銷與投放Agent系統。敘述：基於 CrewAI 與 LangChain 框架開發的智慧行銷引擎。系統以 AI Agent 為核心，自動拆解行銷任務：從即時搜尋競品情報，到利用大模型生成專業文案與視覺素材。用戶僅需提供基礎素材，即可享受「分析、創作、投放」三位一體的自動化服務。"
    # user_input = "產品：可不可熟成紅茶-半熟烏龍厚乳。產品敘述：半熟烏龍厚乳的焙香底蘊融入獨家乳感比例，口感輕盈卻有濃郁乳香，減少奶的厚重，凸顯查的清晰。一口中有茶的輕透與奶的綿密，比奶茶更輕盈，比純茶更柔和，是專屬冬日的舒適厚度。"
    # user_input = "產品：Switch 2。產品敘述：一款專為遊戲玩家設計的高性能遊戲主機，搭載最新的處理器和圖形技術，提供流暢的遊戲體驗。它支持4K解析度和高幀率，並且具有強大的線上功能和豐富的遊戲庫。Switch 2.0 的設計注重便攜性和多功能性，讓玩家可以隨時隨地享受遊戲樂趣。"
    user_input = "玉山銀行熊本熊信用卡"

    inputs = {
        "product_description": user_input,
    }

    try:
        result = CompetitiveAnalysisCrew().crew().kickoff(inputs=inputs)

        print("\n" + "=" * 50)
        print("Competitive analysis complete!")
        print("=" * 50 + "\n")
        print(result)

        with open("competitive_report.md", "w", encoding="utf-8") as f:
            f.write(str(result))

        print(f"\nReport saved to: {os.path.abspath('competitive_report.md')}")

    except Exception as e:
        print(f"\nError: {e}")
        print("Please check your API keys in .env file.")


if __name__ == "__main__":
    run()
