FROM golang:1.26-alpine AS builder

WORKDIR /app

COPY go.mod ./
COPY go.sum* ./
RUN go mod download

COPY . .

RUN go build -o /app/control-plane .

# Deployment Stage
FROM alpine:3.18

WORKDIR /app

COPY --from=builder /app/control-plane /app/control-plane

EXPOSE 8000

CMD ["/app/control-plane"]
