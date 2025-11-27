import os
import uuid
from datetime import datetime

from flask import Flask, jsonify, render_template, request, session

from config.constants import CHROMA_DIR, UPLOAD_DIR
from config.settings import settings
from core.document_processor import DocumentProcessor
from core.rag_engine import RAGEngine

rag_engine = RAGEngine()


def create_app():
    app = Flask(__name__)

    app.config.update(
        SECRET_KEY=settings.SECRET_KEY,
        DEBUG=settings.DEBUG,
        MAX_CONTENT_LENGTH=16 * 1024 * 1024,
        UPLOAD_FOLDER=str(UPLOAD_DIR),
        SESSION_PERMANENT=False,
    )

    # Ensure upload directory exists
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    @app.route("/")
    def index():
        # Initialize session if not exists
        if "session_id" not in session:
            session["session_id"] = str(uuid.uuid4())
            session["chat_history"] = []
            session["uploaded_files"] = []

        return render_template("index.html")

    @app.route("/health")
    def health_check():
        return jsonify(
            {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "environment": settings.FLASK_ENV,
            }
        )

    @app.route("/api/chat", methods=["POST"])
    def chat():
        try:
            data = request.get_json()
            message = data.get("message", "").strip()

            if not message:
                return jsonify({"error": "Message is required"}), 400

            rag_response = rag_engine.query(
                question=message, session_id=session.get("session_id")
            )

            response = {
                "message": rag_response.get["answer"],
                "sources": rag_response.get("sources", []),
                "context_used": rag_response.get("context_used", False),
                "timestamp": datetime.utcnow().isoformat(),
                "session_id": session.get("session_id"),
                "success": rag_response.get("success", True),
            }

            # Add error if present
            if not rag_response.get("success"):
                response["error"] = rag_response.get("error")

            return jsonify(response)

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/upload", methods=["POST"])
    def upload_files():
        """Handle file uploads"""
        try:
            if "files" not in request.files:
                return jsonify({"error": "No files provided"}), 400

            files = request.files.getlist("files")
            uploaded_files = []
            file_paths = []

            for file in files:
                if file.filename == "":
                    continue

                # save files
                filename = f"{session['session_id']}_{file.filename}"
                file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(file_path)

                uploaded_files.append(
                    {
                        "filename": file.filename,
                        "saved_path": file_path,
                        "size": os.path.getsize(file_path),
                    }
                )

            # RAG engine document process
            if file_paths:
                process_result = rag_engine.process_uploaded_documents(file_paths)

                if not process_result.get("success"):
                    return jsonify(
                        {
                            "error": f"Document processing failed: {process_result.get('error')}",
                            "files": uploaded_files,
                        }
                    ), 400

            # Update session
            if "uploaded_files" not in session:
                session["uploaded_files"] = []

            session["uploaded_files"].extend([f["filename"] for f in uploaded_files])
            session.modified = True

            return jsonify(
                {
                    "message": f"Successfully uploaded {len(uploaded_files)} files",
                    "files": uploaded_files,
                }
            )

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/session/clear", methods=["POST"])
    def clear_session():
        """Clear current session and uploaded files"""
        try:
            # Clear uploaded files for this session
            if "uploaded_files" in session:
                for filename in session["uploaded_files"]:
                    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                    if os.path.exists(file_path):
                        os.remove(file_path)

            # Reset session
            session.clear()
            session["session_id"] = str(uuid.uuid4())
            session["chat_history"] = []
            session["uploaded_files"] = []

            return jsonify({"message": "Session cleared successfully"})

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({"error": "Endpoint not found"}), 404

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({"error": "Internal server error"}), 500

    return app


app = create_app()

if __name__ == "__main__":
    print(f"Starting Flask app in {settings.FLASK_ENV} mode")
    print(f"Gemini API Key: {'Set' if settings.GEMINI_API_KEY else 'Not set'}")
    print(f"Web Search: {'Enabled' if settings.ENABLE_WEB_SEARCH else 'Disabled'}")

    app.run(host="0.0.0.0", port=5000, debug=settings.DEBUG)
