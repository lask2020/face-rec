FROM golang:1.22-alpine AS builder

WORKDIR /app

COPY go.mod ./
COPY go.sum* ./
RUN go mod download

COPY main.go ./

# No external dependencies needed — just HTTP client + Redis
RUN CGO_ENABLED=0 go build -o /app/go-worker main.go

# --- Deployment Stage (tiny ~5MB image) ---
FROM alpine:3.19

RUN apk add --no-cache ca-certificates

WORKDIR /app

COPY --from=builder /app/go-worker /app/go-worker

CMD ["/app/go-worker"]
