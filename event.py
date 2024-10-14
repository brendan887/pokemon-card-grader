from flask import Flask, request, jsonify
import hashlib

app = Flask(__name__)


# Set your verification token and endpoint URL
VERIFICATION_TOKEN = "haksjdhahksdghsagdhasdjhasdhashm"
ENDPOINT_URL = "https://shaoqian.sytes.net/ebay/webhook"


@app.route("/ebay/webhook", methods=["GET"])
def ebay_challenge():
    try:
        # Get the challenge code from the query parameters
        challenge_code = request.args.get("challenge_code")
        print(challenge_code)
        if not challenge_code:
            raise ValueError("Missing challenge_code")

        # Concatenate challenge code, verification token, and endpoint URL
        data_to_hash = challenge_code + VERIFICATION_TOKEN + ENDPOINT_URL

        # Compute the SHA-256 hash
        hash_object = hashlib.sha256(data_to_hash.encode())
        challenge_response = hash_object.hexdigest()

        # Respond with the challenge response in JSON format
        response = jsonify({"challengeResponse": challenge_response})
        response.headers["Content-Type"] = "application/json"
        return response, 200
    except Exception as e:
        return jsonify({"error": "Error processing challenge code"}), 500


@app.route("/ebay/webhook", methods=["POST"])
def ebay_webhook():
    try:
        # Get the JSON data from the request
        data = request.json

        # Get the verification token from headers
        verification_token = request.headers.get("X-EBAY-VERIFICATION-TOKEN")

        # Verify the event (optional, implement your verification logic here)
        # verify_ebay_event(data)

        # Process the event (implement your processing logic here)
        process_ebay_event(data)

        # Respond with a 200 status to acknowledge receipt
        return jsonify({"message": "Notification received"}), 200
    except Exception as e:
        return jsonify({"error": "Error processing notification"}), 500


def process_ebay_event(data):
    """
    Process the eBay event notification.
    """
    # Implement your event processing logic here
    event_type = data.get("eventType")
    event_payload = data.get("eventPayload")

    # Add your custom processing logic based on event type
    if event_type == "ITEM_LISTED":
        handle_item_listed(event_payload)
    elif event_type == "ORDER_COMPLETED":
        handle_order_completed(event_payload)
    elif event_type == "MARKETPLACE_ACCOUNT_DELETION":
        handle_marketplace_account_deletion(event_payload)
    # Add more event types as needed


def handle_item_listed(payload):
    """
    Handle the ITEM_LISTED event.
    """
    item_id = payload.get("itemId")
    title = payload.get("title")
    price = payload.get("price")


def handle_order_completed(payload):
    """
    Handle the ORDER_COMPLETED event.
    """
    order_id = payload.get("orderId")
    buyer = payload.get("buyer")
    total = payload.get("total")


def handle_marketplace_account_deletion(payload):
    """
    Handle the MARKETPLACE_ACCOUNT_DELETION event.
    """
    account_id = payload.get("accountId")
    deletion_reason = payload.get("reason")


# Add more handlers for different event types as needed

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=443,
        debug=True,
        ssl_context=(
            "/etc/letsencrypt/live/shaoqian.sytes.net/fullchain.pem",
            "/etc/letsencrypt/live/shaoqian.sytes.net/privkey.pem",
        ),
    )
